"""Prepare a local dense Qwen safetensors checkpoint for MXFP6 weight-free export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from QEfficient.exporter.weight_free import prepare_mxfp6_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path, help="Local Qwen directory containing safetensors and config.json")
    parser.add_argument("output_dir", type=Path, help="New directory for the prepared checkpoint")
    parser.add_argument("--target-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--row-chunk-size", type=int, default=128)
    args = parser.parse_args()
    target_dtype = torch.bfloat16 if args.target_dtype == "bfloat16" else torch.float16
    prepared = prepare_mxfp6_checkpoint(
        args.source_dir,
        args.output_dir,
        target_dtype=target_dtype,
        row_chunk_size=args.row_chunk_size,
    )
    print(prepared)


if __name__ == "__main__":
    main()
