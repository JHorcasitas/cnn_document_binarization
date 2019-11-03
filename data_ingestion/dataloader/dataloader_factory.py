import torch
from data_ingestion.dataloader.dataloader import BinaryDataLoader

from utils.weights import get_weights

Dataset = torch.utils.data.Dataset


def get_dataloader(dataset: Dataset,
                   num_workers: int = 2,
                   batch_size: int = 256) -> BinaryDataLoader:
    dataloader = BinaryDataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  weights=get_weights(dataset))
    return dataloader
