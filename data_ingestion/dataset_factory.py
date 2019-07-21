from .dataset import BinaryDataset


class DatasetFactory:
    @staticmethod
    def get_dataset(kind, transform=None):
        allowed_values = {'train', 'test', 'val'}
        if kind not in allowed_values:
            raise ValueError(f'Kind Value: \'{kind}\' not supported')
        return BinaryDataset(kind=kind, transform=transform)