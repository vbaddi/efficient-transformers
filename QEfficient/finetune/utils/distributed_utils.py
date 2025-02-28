# -----------------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

import torch
import torch.distributed as dist

from ..configs import TrainConfig


def setup_distributed_training(config: TrainConfig) -> None:
    if not config.enable_ddp:
        return
    torch_device = torch.device(config.device)
    assert torch_device.type != "cpu", "Host doesn't support single-node DDP"
    assert torch_device.index is None, f"DDP requires only device type, got: {torch_device}"
    dist.init_process_group(backend=config.dist_backend)
    getattr(torch, torch_device.type).set_device(dist.get_rank())


def cleanup_distributed_training(config: TrainConfig) -> None:
    if config.enable_ddp:
        dist.destroy_process_group()
