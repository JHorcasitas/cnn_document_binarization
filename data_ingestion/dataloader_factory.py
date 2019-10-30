from typing import List

import torch
from .datloader import BinaryDataLoader


Dataset = torch.utils.data.Dataset


# TODO: add support for validation data loader
def get_dataloader(dataset: Dataset,
                   weights: List[float],
                   batch_size: int = 256,
                   num_workers: int = 2) -> BinaryDataLoader:
    dataloader = BinaryDataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  weights=weights)
    return dataloader
