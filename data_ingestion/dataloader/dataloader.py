from itertools import cycle
from math import ceil, floor
from itertools import accumulate
from typing import Optional, List

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler


MAX_DATASET_SIZE = 2 ** 24  # Introduced in Pytorch 1.3.0

Dataset = torch.utils.data.Dataset
Sampler = torch.utils.data.Sampler


class BinaryDataLoader:
    """
    With the introduction of PyTorch 1.3.0 a RuntimeError is thrown when trying
    to iterate with DataLoader over datasets larger than 2 ** 24. This class
    creates n smaller DataLoaders created from n differents partition of the
    datasets and sequentially retreives elements from them
    """
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 256,
                 num_workers: int = 2,
                 weights: Optional[List[float]] = None) -> None:
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._datasets = self._split_dataset(dataset)
        self._weights = self._split_weights(weights)
        self._dataloaders = self._construct_dataloaders()

    def __iter__(self):
        self._c = cycle(range(len(self._datasets)))
        return self

    def __next__(self):
        for i in range(len(self._datasets)):
            try:
                return next(self._dataloaders[next(self._c)])
            except StopIteration:
                continue
        raise StopIteration

    def _split_dataset(self, dataset: Dataset) -> List[Dataset]:
        n_parts = ceil(len(dataset) / MAX_DATASET_SIZE)
        lengths = [floor(len(dataset) / n_parts) for _ in range(n_parts)]
        lengths[-1] += len(dataset) - sum(lengths)
        cum_lengths = [0] + list(accumulate(lengths))
        datasets = []
        for start, offset in zip(cum_lengths, lengths):
            indices = [start + i for i in range(offset)]
            datasets.append(Subset(dataset, indices))
        return datasets

    def _split_weights(self, weights):
        lengths = [len(dataset) for dataset in self._datasets]
        cum_lengths = [0] + list(accumulate(lengths))
        new_weights = []
        for start, offset in zip(cum_lengths, lengths):
            new_weights.append(weights[start: start + offset])
        return new_weights

    def _construct_dataloaders(self):
        dataloaders = []
        for dataset, weights in zip(self._datasets, self._weights):
            sampler = WeightedRandomSampler(weights, len(weights))
            dataloaders.append(DataLoader(dataset=dataset,
                                          sampler=sampler,
                                          num_workers=self._num_workers,
                                          batch_size=self._batch_size))
        return [iter(loader) for loader in dataloaders]
