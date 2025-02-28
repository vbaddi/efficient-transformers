# -----------------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

import fire

from QEfficient.finetune.configs import LoraConfig, TrainConfig
from QEfficient.finetune.finetune_core import finetune_model
from QEfficient.finetune.utils.config_utils import load_config_file, update_config, validate_config


def main(
    model_name: str = None,
    tokenizer_name: str = None,
    batch_size_training: int = None,
    lr: float = None,
    peft_config_file: str = None,
    **kwargs,
) -> None:
    """
    CLI entry point for fine-tuning a model.

    Args:
        model_name (str, optional): Override default model name.
        tokenizer_name (str, optional): Override default tokenizer name.
        batch_size_training (int, optional): Override default training batch size.
        lr (float, optional): Override default learning rate.
        peft_config_file (str, optional): Path to YAML/JSON file containing PEFT (LoRA) config.
        **kwargs: Additional arguments to override TrainConfig.
    """
    train_config = TrainConfig()
    update_config(train_config, **kwargs)

    lora_config = LoraConfig()
    if peft_config_file:
        peft_config_data = load_config_file(peft_config_file)
        validate_config(peft_config_data, config_type="lora")
        lora_config = LoraConfig(**peft_config_data)

    finetune_model(train_config, lora_config)


if __name__ == "__main__":
    fire.Fire(main)
