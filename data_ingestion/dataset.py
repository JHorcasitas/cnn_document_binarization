import os

from PIL import Image
from torch.utils.data import Dataset

import config


class BinaryDataset(Dataset):

    def __init__(self, kind, input_transform=None, target_transform=None):
        """
        args:
        kind (str): one of: {train, test, val}
        transform (): transformation to apply to every image
        """
        self._kind = kind
        self._input_transform  = input_transform
        self._target_transform = target_transform

        cfg = config.read_config()
        self._radius     = cfg['radius']
        self._data_path  = cfg['data_path']

        self._input_path  = os.path.join(self._data_path, 'input', self._kind)
        self._target_path = os.path.join(self._data_path, 'target', self._kind)

        order = lambda x: int(os.path.splitext(x)[0])
        self._input_names  = sorted(os.listdir(self._input_path), key=order)
        self._target_names = sorted(os.listdir(self._target_path), key=order)

        if len(self._input_names) != len(self._target_names):
            msg = (f'Inconsisten number of files between: {self._input_path} '
                   f'and {self._target_path}')
            raise AssertionError(msg)

    def __len__(self):
        return len(self._input_names)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self._input_path, self._input_names[idx])
        input = Image.open(input_path).convert('RGB')

        target_path = os.path.join(self._target_path, self._target_names[idx])
        target = Image.open(target_path).convert('1')

        if self._input_transform:
            input = self._input_transform(input)
        if self._target_transform:
            target = self._target_transform(target)

        return input, target