# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch

from ..dataset import DATALOADER_COLLATE_FUNC, DATASET_PREPROC


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train", context_length: int = None
) -> torch.utils.data.Dataset:
    if dataset_config.dataset not in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not implemented")
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        dataset_config.train_split if split == "train" else dataset_config.test_split,
        context_length,
    )


def get_custom_data_collator(dataset_processer, dataset_config) -> torch.utils.data.Dataset:
    return DATALOADER_COLLATE_FUNC.get(dataset_config.dataset, lambda x, y: None)(dataset_processer, dataset_config)


def setup_dataloaders(train_config, dataset_config, tokenizer, dataset_train, dataset_val):
    from .config_utils import get_dataloader_kwargs

    custom_data_collator = get_custom_data_collator(tokenizer, dataset_config)
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")
    if custom_data_collator:
        train_dl_kwargs["collate_fn"] = custom_data_collator
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")
    eval_dataloader = None
    if train_config.run_validation:
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError("Eval set too small to load even one batch.")
    return train_dataloader, eval_dataloader
