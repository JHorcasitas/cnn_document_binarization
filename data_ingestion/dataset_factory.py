from .dataset import BinaryDataset
from .dataset import StridedDataset


def get_dataset(dataset,
                kind,
                input_transform=None,
                target_transform=None):
    """
    Retrieves especified dataset
    Args:
        dataset (str): either ´binary´ or ´strided´
        kind (str): one of {train, test, val}
        input_transform (callable): transformation to apply on input
        target_transform (callable): transformation to apply on target
    """
    if kind not in {'train', 'test', 'val'}:
        raise ValueError(f'Kind value: \'{kind}\' not supported')

    if dataset == 'binary':
        return BinaryDataset(kind=kind,
                             input_transform=input_transform)
    elif dataset == 'strided':
        return StridedDataset(kind=kind,
                              input_transform=input_transform,
                              target_transform=target_transform)
    else:
        raise ValueError(f'Dataset value: \'{dataset}\' not supported')
