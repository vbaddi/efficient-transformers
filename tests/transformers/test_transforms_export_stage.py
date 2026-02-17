#!/usr/bin/env python3
# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner

TEST_CONFIGS: List[Tuple[str, int, int, int, int, int, int, Dict]] = [
    ("gpt2", 256, 2, 4, 128, 512, 127, {}),
    ("codegen", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("falcon", 256, 2, 4, 128, 512, 127, {}),
    ("gptj", 256, 2, 4, 128, 512, 127, {"rotary_dim": 16}),
    ("llama", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mistral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mixtral", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("mpt", 256, 2, 4, 128, 512, 127, {}),
    ("phi", 256, 2, 4, 128, 512, 127, {}),
    ("phi3", 256, 2, 4, 128, 512, 127, {"pad_token_id": 0}),
    ("qwen2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("starcoder2", 256, 2, 4, 128, 512, 127, {}),
    ("granite", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("olmo2", 256, 2, 4, 128, 512, 127, {"num_key_value_heads": 2}),
    ("gpt_oss", 256, 3, 4, 128, 512, 127, {"num_key_value_heads": 2}),
]


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    unk_token_id = 3
    pad_token = "<pad>"
    eos_token = "<eos>"
    bos_token = "<bos>"
    unk_token = "<unk>"
    padding_side = "right"
    model_max_length = 1024

    def __len__(self):
        return 128

    def __call__(self, prompt, return_tensors="pt", padding=True):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        token_rows = []
        for text in prompts:
            token_ids = [4 + (ord(ch) % 17) for ch in text[:6]]
            if not token_ids:
                token_ids = [4, 5]
            token_rows.append(token_ids)

        max_len = max(len(ids) for ids in token_rows)
        arr = np.full((len(token_rows), max_len), self.pad_token_id, dtype=np.int64)
        mask = np.zeros_like(arr)
        for i, ids in enumerate(token_rows):
            arr[i, : len(ids)] = np.array(ids, dtype=np.int64)
            mask[i, : len(ids)] = 1

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(arr, dtype=torch.int64),
                "attention_mask": torch.tensor(mask, dtype=torch.int64),
            }
        if return_tensors == "np":
            return {"input_ids": arr, "attention_mask": mask}
        raise ValueError(f"Unsupported return_tensors={return_tensors}")

    def encode(self, text, return_tensors="pt"):
        ids = [4 + (ord(ch) % 17) for ch in text[:6]]
        if not ids:
            ids = [4, 5]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.int64)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return " ".join(map(str, ids))

    def batch_decode(self, batch_ids, skip_special_tokens=True):
        if isinstance(batch_ids, np.ndarray):
            return [self.decode(row.tolist(), skip_special_tokens=skip_special_tokens) for row in batch_ids]
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in batch_ids]


def run_config_smoke(selected_models: List[str], use_onnx_subfunctions: bool):
    tokenizer = TinyTokenizer()
    rows = []

    for model_name, max_pos, n_layers, n_heads, hidden, inter, vocab, extra in TEST_CONFIGS:
        if selected_models and model_name not in selected_models:
            continue

        print(f"\n=== {model_name} ===")
        try:
            config = AutoConfig.for_model(
                model_name,
                max_position_embeddings=max_pos,
                num_hidden_layers=n_layers,
                num_attention_heads=n_heads,
                hidden_size=hidden,
                intermediate_size=inter,
                vocab_size=vocab,
                **extra,
            )
            model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")

            runner = ApiRunner(
                batch_size=1,
                tokenizer=tokenizer,
                config=config,
                prompt=["My name is"],
                prompt_len=8,
                ctx_len=32,
            )

            qeff_model = QEFFAutoModelForCausalLM(model)
            # For deferred-transform flow validation: ensure PT run uses transformed model.
            qeff_model._apply_pytorch_transforms()
            pt_tokens = runner.run_kv_model_on_pytorch(qeff_model.model)

            onnx_path = qeff_model.export(
                use_dynamo=True,
                use_onnx_subfunctions=use_onnx_subfunctions,
                offload_pt_weights=False,
            )
            ort_tokens = runner.run_kv_model_on_ort(onnx_path)
            match = np.array_equal(pt_tokens, ort_tokens)
            print(f"match={match}")

            rows.append((model_name, bool(match), "match" if match else "mismatch"))
        except Exception as exc:
            print(f"FAILED: {exc}")
            traceback.print_exc()
            rows.append((model_name, False, str(exc)))

    print("\n=== SUMMARY ===")
    for name, ok, msg in rows:
        print(f"{name:12} {'PASS' if ok else 'FAIL'} {msg}")
    failed = [row for row in rows if not row[1]]
    print(f"total={len(rows)} passed={len(rows) - len(failed)} failed={len(failed)}")
    return 0 if not failed else 1


def parse_args():
    parser = argparse.ArgumentParser(description="Config-only smoke test for QEFF CausalLM PT-KV vs ORT-KV parity.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Optional subset of model names from TEST_CONFIGS (e.g., gpt2 llama).",
    )
    parser.add_argument(
        "--no-subfunctions",
        action="store_true",
        help="Disable ONNX subfunction export path (default: enabled).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    return run_config_smoke(
        selected_models=args.models,
        use_onnx_subfunctions=not args.no_subfunctions,
    )


if __name__ == "__main__":
    raise SystemExit(main())
