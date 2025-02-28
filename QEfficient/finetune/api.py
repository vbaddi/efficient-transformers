# -----------------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

from typing import Union

from .configs import LoraConfig, TrainConfig
from .finetune_core import finetune_model
from .utils.config_utils import load_config_file, validate_config


def finetune(train_config: TrainConfig, lora_config: Union[LoraConfig, str]):
    """
    API entry point for fine-tuning a model programmatically.

    Args:
        train_config (TrainConfig): Training configuration object.
        lora_config (Union[LoraConfig, str]): LoRA configuration object or path to config file.

    Returns:
        dict: Results dictionary from the training process.
    """
    if isinstance(lora_config, str):
        peft_config_data = load_config_file(lora_config)
        validate_config(peft_config_data, config_type="lora")
        lora_config = LoraConfig(**peft_config_data)
    elif not isinstance(lora_config, LoraConfig):
        raise ValueError("lora_config must be a LoraConfig object or a path to a config file.")

    return finetune_model(train_config, lora_config)
