# -----------------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

import torch
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...utils._utils import login_and_download_hf_lm
from ..configs import LoraConfig, TrainConfig
from ..utils.config_utils import generate_peft_config


def load_model_and_tokenizer(config: TrainConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    pretrained_model_path = login_and_download_hf_lm(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        use_cache=False,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name if config.tokenizer_name is None else config.tokenizer_name
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing embedding matrix to match tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def apply_peft(model: AutoModelForCausalLM, train_config: TrainConfig, lora_config: LoraConfig) -> PeftModel:
    if not train_config.use_peft:
        return model
    if train_config.from_peft_checkpoint:
        return PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
    peft_config = generate_peft_config(train_config, lora_config)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
