# -----------------------------------------------------------------------------
#
# Copyright (c)  2023-2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import logging
import os
from typing import List, Optional

import QEfficient
from QEfficient.cloud.export import get_onnx_model_path
from QEfficient.generation.text_generation_inference import (
    check_batch_size_and_num_prompts,
    cloud_ai_100_exec_kv,
)
from QEfficient.utils import check_and_assign_cache_dir, get_qpc_dir_path, load_hf_tokenizer, qpc_exists
from QEfficient.utils.logging_utils import logger

"""
1. Check if compiled qpc for given config already exists, if it does jump to execute, else
2. Check if exported ONNX file already exists, if true, jump to compilation -> execution, else
3. Check if HF model exists in cache, if true, start transform -> export -> compilation -> execution, else,
4. Download HF model -> transform -> export -> compile -> execute
"""


def main(
    model_name: str,
    num_cores: int,
    prompt: Optional[str] = None,  # type: ignore
    local_model_dir: Optional[str] = None,
    prompts_txt_file_path: Optional[str] = None,
    aic_enable_depth_first: bool = False,
    mos: int = -1,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    batch_size: int = 1,
    prompt_len: int = 32,
    ctx_len: int = 128,
    generation_len: Optional[int] = None,
    mxfp6: bool = False,
    mxint8: bool = False,
    device_group: List[int] = [
        0,
    ],
) -> None:
    prompt: List[str] = check_batch_size_and_num_prompts(prompt, prompts_txt_file_path, batch_size)
    cache_dir = check_and_assign_cache_dir(local_model_dir, cache_dir)

    tokenizer = load_hf_tokenizer(
        pretrained_model_name_or_path=(local_model_dir if local_model_dir else model_name),
        cache_dir=cache_dir,
        hf_token=hf_token,
        local_model_dir=local_model_dir,
    )

    qpc_dir_path = get_qpc_dir_path(
        model_name, num_cores, mos, batch_size, prompt_len, ctx_len, mxfp6, mxint8, device_group
    )

    # Handle qpc generation
    if qpc_exists(qpc_dir_path):
        logger.info(f"Pre-compiled qpc found at {qpc_dir_path}! Executing with given prompt")
    else:
        # Handle onnx model generation
        onnx_model_path = get_onnx_model_path(model_name, cache_dir, tokenizer, hf_token, local_model_dir)

        #########
        # Compile
        #########
        generated_qpc_path = QEfficient.compile(
            onnx_path=onnx_model_path,
            qpc_path=os.path.dirname(
                qpc_dir_path
            ),  # We need to pass parent directory of qpc_dir_path, as the compile function handles the qpcs directory creation
            num_cores=num_cores,
            batch_size=batch_size,
            prompt_len=prompt_len,
            ctx_len=ctx_len,
            mxfp6=mxfp6,
            mxint8=mxint8,
            aic_enable_depth_first=aic_enable_depth_first,
            mos=mos,
            device_group=device_group,
        )
        assert (
            generated_qpc_path == qpc_dir_path
        ), f"QPC files were generated at an unusual location, expected {qpc_dir_path}; got {generated_qpc_path}"

    #########
    # Execute
    #########
    cloud_ai_100_exec_kv(
        batch_size,
        tokenizer=tokenizer,
        qpc_path=qpc_dir_path,
        device_id=device_group,
        prompt=prompt,
        ctx_len=ctx_len,
        generation_len=generation_len,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference command, the model will be downloaded from HF, optmized, compiled, executed on Cloud AI 100"
    )
    parser.add_argument("--model-name", "--model_name", required=True, help="HF Model card name/id")
    parser.add_argument(
        "--local-model-dir", "--local_model_dir", required=False, help="Path to custom model weights and config files"
    )
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        required=False,
        help="Cache dir to store HF Downloads",
    )
    parser.add_argument(
        "--hf-token", "--hf_token", default=None, type=str, required=False, help="HF token id for private HF models"
    )
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1, help="Batch size for text generation")
    parser.add_argument(
        "--prompt-len", "--prompt_len", default=32, type=int, help="Sequence length for text generation."
    )
    parser.add_argument("--ctx-len", "--ctx_len", default=128, type=int, help="Context length for text generation.")
    parser.add_argument(
        "--mxfp6", action="store_true", help="Compress constant MatMul weights to MXFP6 E2M3, default is no compression"
    )
    parser.add_argument(
        "--mxint8",
        action="store_true",
        help="Compress Present/Past KV to MXINT8 using CustomIO config, default is False",
    )
    parser.add_argument(
        "--num_cores", "--num-cores", type=int, required=True, help="Number of cores to compile on Cloud AI 100"
    )
    parser.add_argument(
        "--device_group",
        "--device-group",
        required=True,
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        help="Cloud AI 100 device ids (comma-separated) e.g. [0,1]  ",
    )
    parser.add_argument(
        "--prompt",
        type=lambda prompt: prompt.split("|"),
        help="Input prompt, if executing for batch size>1, pass input prompts in single string but seperate with pipe (|) symbol",
    )
    parser.add_argument(
        "--prompts_txt_file_path",
        "--prompts-txt-file-path",
        type=str,
        help="File path for taking input prompts from txt file, sample prompts.txt file present in examples folder",
    )
    parser.add_argument("--generation_len", "--generation-len", type=int, help="Number of tokens to generate")
    parser.add_argument(
        "--aic_enable_depth_first",
        "--aic-enable-depth-first",
        action="store_true",
        help="If passed, this option will be enabled during compilation, disabled by default",
    )
    parser.add_argument(
        "--mos",
        type=int,
        default=-1,
        help="Effort level to reduce the on-chip memory",
    )
    # FIXME: Add verbose feature
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="pass to print info logs",
    )

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    del args.verbose  # type: ignore
    main(**args.__dict__)
