import os
import pandas as pd
from torch.utils.data import Dataset


class BinaryDataset(Dataset):

    def __init__(self, data_path, labels_path, transform=None):
        """
        args:
        data_path (str): path where images are located
        labels_path (str): path where labels are located
        transform (): transformation to apply to every image
        """
        self._data_path = data_path
        self._labels_path = labels_path
        self._transform = transform
        self.validate_paths()
    
    def validate_paths(self):
        if os.path.isdir(self._data_path):
            msg = f'No such directory: \'{self._data_path}\''
            raise FileNotFoundError(msg)
        if os.path.isdir(self._labels_path):
            msg = f'No such directory: \'{self._labels_path}\''
            raise FileNotFoundError(msg)
        
        num_file_data    = len(os.listdir(self._data_path))
        num_files_labels = len(os.listdir(self._labels_path))
        if num_file_data != num_files_labels:
            msg = (f'Different number of files between {self._data_path} and'
                   f' {self._labels_path}')
            raise AssertionError(msg)

    def __len__(self):
        return num_file_data 