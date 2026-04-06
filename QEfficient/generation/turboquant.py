# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


@dataclass
class TurboQuantCompressionStats:
    stage: str
    seq_len: int
    n_layers: int
    n_heads: int
    head_dim: int
    fp16_bytes: float
    turbo_bytes: float
    ratio: float
    enabled: bool
    reason: str

    @property
    def fp16_mb(self) -> float:
        return self.fp16_bytes / (1024 * 1024)

    @property
    def turbo_mb(self) -> float:
        return self.turbo_bytes / (1024 * 1024)


class TurboQuantStatsTracker:
    """
    Computes TurboQuant KV-cache compression stats for a QAIC run.

    Notes:
    - This is a host-side estimator based on the published TurboQuant packing model.
    - Cloud AI 100 retained-state memory format remains unchanged in this integration.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        batch_size: int,
        total_bits: int = 3,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.batch_size = batch_size
        self.total_bits = total_bits
        self.mse_bits = max(total_bits - 1, 1)

    def is_supported(self) -> tuple[bool, str]:
        if self.head_dim != 128:
            return False, f"TurboQuant currently expects head_dim=128, got {self.head_dim}"
        if self.total_bits not in (2, 3, 4):
            return False, f"Unsupported total_bits={self.total_bits}"
        if self.n_layers <= 0 or self.n_heads <= 0:
            return False, "Unable to infer valid KV cache geometry"
        return True, "ok"

    def _compressed_bytes_per_head_layer(self, seq_len: int) -> float:
        # Matches turboquant-gpu host-side size accounting:
        # key bytes  = (seq*d*mse_bits + seq*d*(qjl_sign) + seq*32(norms)) / 8
        # value bytes= (seq*d*total_bits + seq*16(norms)) / 8
        d = self.head_dim
        key_bytes = (seq_len * d * self.mse_bits + seq_len * d + seq_len * 32) / 8
        value_bytes = (seq_len * d * self.total_bits + seq_len * 16) / 8
        return key_bytes + value_bytes

    def estimate(self, stage: str, seq_len: int) -> TurboQuantCompressionStats:
        supported, reason = self.is_supported()
        seq_len = max(int(seq_len), 1)

        fp16_bytes = float(self.batch_size * self.n_layers * self.n_heads * seq_len * self.head_dim * 2 * 2)
        turbo_bytes = float(
            self.batch_size * self.n_layers * self.n_heads * self._compressed_bytes_per_head_layer(seq_len)
        )
        ratio = fp16_bytes / turbo_bytes if turbo_bytes > 0 else 1.0
        if not supported:
            turbo_bytes = fp16_bytes
            ratio = 1.0

        return TurboQuantCompressionStats(
            stage=stage,
            seq_len=seq_len,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            head_dim=self.head_dim,
            fp16_bytes=fp16_bytes,
            turbo_bytes=turbo_bytes,
            ratio=ratio,
            enabled=supported,
            reason=reason,
        )

    @staticmethod
    def format_report(stats: TurboQuantCompressionStats) -> str:
        line = "=" * 69
        status = "enabled" if stats.enabled else f"disabled ({stats.reason})"
        return (
            f"\n{line}\n"
            f"TurboQuant KV Compression Estimate ({status})\n"
            f"stage={stats.stage} | seq_len={stats.seq_len} | layers={stats.n_layers} | heads={stats.n_heads} | head_dim={stats.head_dim}\n"
            f"KV FP16 size     : {stats.fp16_mb:.2f} MB\n"
            f"KV TurboQuant size: {stats.turbo_mb:.2f} MB\n"
            f"compression ratio: {stats.ratio:.2f}x\n"
            f"{line}\n"
        )


def _rotation_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    g = torch.randn(d, d, generator=gen)
    q, r = torch.linalg.qr(g)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign[diag_sign == 0] = 1.0
    return (q * diag_sign.unsqueeze(0)).to(device=device, dtype=torch.float32).contiguous()


def _qjl_matrix(d: int, seed: int, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 10000)
    return torch.randn(d, d, generator=gen).to(device=device, dtype=torch.float32).contiguous()


class _GaussianCodebook:
    def __init__(self, d: int, bits: int):
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.sigma = 1.0 / math.sqrt(d)
        self.boundaries, self.centroids = self._build(d, bits)

    @staticmethod
    def _build(d: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_levels = 1 << bits
        sigma = 1.0 / math.sqrt(d)
        eps = 1e-6
        b_probs = torch.linspace(1, n_levels - 1, n_levels - 1, dtype=torch.float32) / n_levels
        b_probs = b_probs.clamp(eps, 1 - eps)
        boundaries = math.sqrt(2.0) * torch.special.erfinv(2 * b_probs - 1) * sigma

        c_probs = (torch.arange(n_levels, dtype=torch.float32) + 0.5) / n_levels
        c_probs = c_probs.clamp(eps, 1 - eps)
        centroids = math.sqrt(2.0) * torch.special.erfinv(2 * c_probs - 1) * sigma
        return boundaries.contiguous(), centroids.contiguous()

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        boundaries = self.boundaries.to(x.device)
        return torch.bucketize(x, boundaries).to(torch.uint8)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        centroids = self.centroids.to(indices.device)
        return centroids[indices.long()]


class TurboQuantPytorchCodec:
    """
    Host-side TurboQuant-like codec (PyTorch implementation).

    This is intended for QAIC host-managed KV experimentation:
    - compresses retained-state KV tensors on host
    - decompresses and feeds them back as `past_key/value.*` inputs
    """

    def __init__(self, head_dim: int = 128, total_bits: int = 3, seed: int = 42):
        self.head_dim = head_dim
        self.total_bits = total_bits
        self.mse_bits = max(total_bits - 1, 1)
        self.pi = _rotation_matrix(head_dim, seed, device="cpu")
        self.pi_t = self.pi.t().contiguous()
        self.s_t = _qjl_matrix(head_dim, seed, device="cpu").t().contiguous()
        self.key_codebook = _GaussianCodebook(head_dim, self.mse_bits)
        self.val_codebook = _GaussianCodebook(head_dim, total_bits)

    def is_supported(self) -> tuple[bool, str]:
        if self.head_dim != 128:
            return False, f"TurboQuant PyTorch codec expects head_dim=128, got {self.head_dim}"
        if self.total_bits not in (2, 3, 4):
            return False, f"Unsupported total_bits={self.total_bits}"
        return True, "ok"

    @staticmethod
    def _pack_bits(values: np.ndarray, bits: int) -> np.ndarray:
        flat = values.reshape(-1).astype(np.uint8)
        total_bits = int(flat.size * bits)
        packed = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
        positions = np.arange(flat.size, dtype=np.int64) * bits
        for bit_idx in range(bits):
            bit_vals = ((flat >> bit_idx) & 1).astype(np.uint8)
            bit_positions = positions + bit_idx
            shifts = (bit_positions % 8).astype(np.uint8)
            packed[bit_positions // 8] |= (bit_vals << shifts).astype(np.uint8)
        return packed

    @staticmethod
    def _unpack_bits(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
        values = np.zeros(count, dtype=np.uint8)
        positions = np.arange(count, dtype=np.int64) * bits
        for bit_idx in range(bits):
            bit_positions = positions + bit_idx
            bit_vals = (packed[bit_positions // 8] >> (bit_positions % 8)) & 1
            values |= (bit_vals.astype(np.uint8) << bit_idx).astype(np.uint8)
        return values

    def _compress_tensor(self, x: np.ndarray, is_key: bool) -> dict:
        t = torch.from_numpy(np.array(x, copy=True)).to(torch.float32)
        orig_shape = t.shape
        flat = t.reshape(-1, orig_shape[-1]).contiguous()
        norms = torch.linalg.norm(flat, dim=-1, keepdim=True)
        normed = flat / (norms + 1e-8)
        rotated = normed @ self.pi_t
        codebook = self.key_codebook if is_key else self.val_codebook
        bits = self.mse_bits if is_key else self.total_bits
        indices = codebook.quantize(rotated).cpu().numpy()
        out = {
            "packed_indices": self._pack_bits(indices, bits=bits),
            "indices_numel": int(indices.size),
            "indices_shape": tuple(indices.shape),
            "bits": bits,
            "norms": norms.squeeze(-1).to(torch.float16).cpu().numpy(),
            "shape": tuple(orig_shape),
            "is_key": is_key,
        }
        if is_key:
            y_hat = codebook.dequantize(torch.from_numpy(np.array(indices, copy=True)).to(torch.uint8))
            k_mse = (y_hat @ self.pi) * norms
            residual = flat - k_mse
            signs = (residual @ self.s_t >= 0).to(torch.uint8).cpu().numpy()
            out["packed_signs"] = self._pack_bits(signs, bits=1)
            out["residual_norms"] = torch.linalg.norm(residual, dim=-1).to(torch.float16).cpu().numpy()
        return out

    def _decompress_tensor(self, comp: dict) -> np.ndarray:
        codebook = self.key_codebook if comp["is_key"] else self.val_codebook
        unpacked = self._unpack_bits(comp["packed_indices"], comp["bits"], comp["indices_numel"])
        unpacked = unpacked.reshape(comp["indices_shape"])
        indices = torch.from_numpy(np.array(unpacked, copy=True)).to(torch.uint8)
        norms = torch.from_numpy(np.array(comp["norms"], copy=True)).to(torch.float32).unsqueeze(-1)
        y_hat = codebook.dequantize(indices)
        out = ((y_hat @ self.pi) * norms).reshape(comp["shape"]).to(torch.float16)
        return out.numpy()

    def compress_cache(self, cache_outputs: Dict[str, np.ndarray]) -> dict:
        compressed = {}
        for out_name, tensor in cache_outputs.items():
            if out_name.startswith("past_key.") and out_name.endswith("_RetainedState"):
                input_name = out_name[: -len("_RetainedState")]
                compressed[input_name] = self._compress_tensor(tensor, is_key=True)
            elif out_name.startswith("past_value.") and out_name.endswith("_RetainedState"):
                input_name = out_name[: -len("_RetainedState")]
                compressed[input_name] = self._compress_tensor(tensor, is_key=False)
        return compressed

    def decompress_cache(self, compressed_cache: dict) -> Dict[str, np.ndarray]:
        return {name: self._decompress_tensor(comp) for name, comp in compressed_cache.items()}

    @staticmethod
    def compressed_cache_bytes(compressed_cache: dict) -> int:
        total = 0
        for comp in compressed_cache.values():
            total += int(comp["packed_indices"].nbytes)
            total += int(comp["norms"].nbytes)
            if comp.get("packed_signs") is not None:
                total += int(comp["packed_signs"].nbytes)
            if comp.get("residual_norms") is not None:
                total += int(comp["residual_norms"].nbytes)
        return total
