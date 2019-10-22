from typing import Optional

import torch
from .datloader import BinaryDataLoader


Dataset = torch.utils.data.Dataset
Sampler = torch.utils.data.Sampler


def get_dataloader(dataset: Dataset,
                   batch_size: int = 256,
                   num_workers: int = 2,
                   shuffle: bool = False,
                   sampler: Optional[Sampler] = None) -> BinaryDataLoader:
    dataloader = BinaryDataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle,
                                  sampler=sampler)
    return dataloader
