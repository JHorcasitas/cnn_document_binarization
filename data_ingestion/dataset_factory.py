from .dataset import BinaryDataset
from .dataset import StridedDataset


def get_dataset(dataset,
                kind,
                input_transform=None,
                target_transform=None):
    
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