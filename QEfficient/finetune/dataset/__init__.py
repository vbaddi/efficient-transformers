# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from functools import partial

from .alpaca_dataset import InstructionDataset as get_alpaca_dataset
from .custom_dataset import get_custom_dataset, get_data_collator
from .grammar_dataset import get_dataset as get_grammar_dataset
from .gsm8k_dataset import get_gsm8k_dataset
from .samsum_dataset import get_preprocessed_samsum as get_samsum_dataset

DATASET_PREPROC = {
    "alpaca_dataset": partial(get_alpaca_dataset),
    "samsum_dataset": get_samsum_dataset,
    "grammar_dataset": get_grammar_dataset,
    "gsm8k_dataset": get_gsm8k_dataset,
    "custom_dataset": get_custom_dataset,
    # Add other datasets here as implemented
}

DATALOADER_COLLATE_FUNC = {"custom_dataset": get_data_collator}

__all__ = [
    "get_samsum_dataset",
    "get_alpaca_dataset",
    "get_grammar_dataset",
    "get_gsm8k_dataset",
    "get_custom_datasetDATASET_PREPROC",
    "DATALOADER_COLLATE_FUNC",
]
