# -----------------------------------------------------------------------------
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .configs import LoraConfig, TrainConfig
from .utils.config_utils import generate_dataset_config
from .utils.dataset_utils import get_preprocessed_dataset, setup_dataloaders
from .utils.distributed_utils import cleanup_distributed_training, setup_distributed_training
from .utils.model_utils import apply_peft, load_model_and_tokenizer
from .utils.seed_utils import setup_seeds
from .utils.train_utils import get_longest_seq_length, print_model_size, train


def finetune_model(train_config: TrainConfig, lora_config: LoraConfig):
    """
    Core function to fine-tune a model based on provided configurations.

    Args:
        train_config (TrainConfig): Training configuration object.
        lora_config (LoraConfig): LoRA configuration object for PEFT.

    Returns:
        dict: Results dictionary from the training process.
    """
    # Setup distributed training if enabled
    setup_distributed_training(train_config)

    # Set random seeds for reproducibility
    setup_seeds(train_config.seed)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(train_config)
    print_model_size(model, train_config)

    # Apply PEFT if enabled
    model = apply_peft(model, train_config, lora_config)

    # Generate dataset configuration
    dataset_config = generate_dataset_config(train_config)

    # Preprocess datasets
    dataset_train = get_preprocessed_dataset(
        tokenizer, dataset_config, split="train", context_length=train_config.context_length
    )
    dataset_val = get_preprocessed_dataset(
        tokenizer, dataset_config, split="test", context_length=train_config.context_length
    )

    # Setup DataLoaders
    train_dataloader, eval_dataloader = setup_dataloaders(
        train_config, dataset_config, tokenizer, dataset_train, dataset_val
    )

    # Compute longest sequence length for logging
    dataset_for_seq_length = (
        torch.utils.data.ConcatDataset([train_dataloader.dataset, eval_dataloader.dataset])
        if train_config.run_validation
        else train_dataloader.dataset
    )
    longest_seq_length, _ = get_longest_seq_length(list(dataset_for_seq_length))
    print(
        f"Longest sequence length: {longest_seq_length}, "
        f"Context length: {train_config.context_length}, "
        f"Model max context: {model.config.max_position_embeddings}"
    )

    # Move model to device
    model.to(train_config.device)

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Wrap model in DDP if enabled
    rank = None
    if train_config.enable_ddp:
        import torch.distributed as dist

        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
        rank = dist.get_rank()

    # Train the model
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        train_config.device,
        rank,
        None,
    )

    # Cleanup distributed training if enabled
    cleanup_distributed_training(train_config)

    return results
