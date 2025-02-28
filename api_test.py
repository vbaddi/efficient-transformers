# -----------------------------------------------------------------------------
#
# Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from QEfficient.finetune import api
from QEfficient.finetune.configs import LoraConfig, TrainConfig

train_config = TrainConfig(model_name="meta-llama/Llama-3.2-1B", lr=5e-4, enable_ddp=False, device="qaic")
lora_config = LoraConfig(r=8, lora_alpha=32)
results = api.finetune(train_config, "/home/vbaddi/finetune_infra/efficient-transformers/lora_config.yaml")
print(results)
