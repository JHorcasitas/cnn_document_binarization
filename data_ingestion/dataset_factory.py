from .dataset import BinaryDataset
from .dataset import StridedDataset


def get_dataset(dataset,
                kind,
                transform_input=False,
                transform_target=False):
    """
    Retrieves especified dataset
    Args:
        dataset (str): either ´binary´ or ´strided´
        kind (str): one of {train, test, val}
        transform_input (bool): wether to transform target so that it can be
        processed by PyTorch
        transform_target (bool): wether to transform target so that it can be
        processed by PyTorch
    """
    if kind not in {'train', 'test', 'val'}:
        raise ValueError(f'Kind value: \'{kind}\' not supported')

    if dataset == 'binary':
        return BinaryDataset(kind=kind,
                             transform_input=transform_input)
    elif dataset == 'strided':
        return StridedDataset(kind=kind,
                              transform_input=transform_input,
                              transform_target=transform_target)
    else:
        raise ValueError(f'Dataset value: \'{dataset}\' not supported')
