from .dataset import BinaryDataset


class DatasetFactory:
    def get_dataset(self, kind, input_transform=None, target_transform=None):
        if kind not in {'train', 'test', 'val'}:
            raise ValueError(f'Kind Value: \'{kind}\' not supported')
        return BinaryDataset(kind=kind,
                             input_transform=input_transform,
                             target_transform=target_transform)