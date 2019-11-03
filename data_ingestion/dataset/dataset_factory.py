from typing import Union
from data_ingestion.dataset.dataset import BinaryDataset, StridedDataset


Dataset = Union[BinaryDataset, StridedDataset]


def get_dataset(dataset: str,
                kind: str,
                transform_input: bool = True,
                transform_target: bool = True) -> Dataset:
    """
    Retrieves the especified dataset
    Args:
        dataset: either 'binary' or 'strided'
        kind: one of {'train', 'test', 'val'}
        transform_input: wether to transform target so that it can be processed
        by PyTorch
        transform_target: wether to transform target so that it can be
        processed by PyTorch
    """
    if kind in {'train', 'val'}:
        if dataset == 'binary':
            return BinaryDataset(kind=kind,
                                 transform_input=transform_input,
                                 transform_target=transform_target)
        elif dataset == 'strided':
            return StridedDataset(kind=kind,
                                  transform_input=transform_input,
                                  transform_target=transform_target)
        else:
            raise ValueError(f'Unsupported dataset value: {dataset}')
    else:
        raise ValueError(f'Unsupported kind value: {kind}')
