import os
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import config


class BinaryDataset(Dataset):

    def __init__(self, kind, transform=None):
        """
        args:
        kind (str): one of: {train, test, val}
        transform (): transformation to apply to every image
        """
        cfg = config.read_config()

        self._kind = kind
        self._radius     = cfg['radius']
        self._data_path  = cfg['data_path']
        self._transform  = transform
        
        self._img_sizes = self._get_img_sizes()
    
    def _get_img_sizes(self):
        img_sizes = OrderedDict()
        path = os.path.join(self._data_path, 'input', self._kind)
        order = lambda x: int(os.path.splitext(x)[0])
        for img_name in sorted(os.listdir(path), key=order):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, 0)
            img_sizes[img_name] = img.shape[0] * img.shape[1]
        return img_sizes

    def __len__(self):
        num_patterns = 0
        for key in self._img_sizes:
            num_patterns += self._img_sizes[key]
        return num_patterns
    
    def __getitem__(self, idx):

        def idx_to_img_name(idx):
            total = 0
            for key in self._img_sizes:
                total += self._img_sizes[key]
                if idx < total:
                    return key, idx - (total - self._img_sizes[key]) + 1
            raise RuntimeError(f'Index {idx} could not be found')
        
        def get_coord_from_idx(input_img, img_idx):
            h, w = input_img.shape
            row = img_idx // w
            col = (img_idx % w) -1
            return col, row

        img_name, img_index = idx_to_img_name(idx)
        print(f'Image name: {img_name}')
        input_path = os.path.join(self._data_path,
                                  'input',
                                  self._kind,
                                  img_name)
        target_path = os.path.join(self._data_path,
                                   'target',
                                   self._kind,
                                   img_name)

        input_img  = cv2.imread(input_path, 0)
        target_img = cv2.imread(target_path, 0)
        target_img = np.where(target_img > 0, 1, 0)
        x, y = get_coord_from_idx(input_img, img_index)
        print(f'x: {x}, y: {y}')

        input_img = np.pad(array=input_img,
                           pad_width=self._radius,
                           mode='constant',
                           constant_values=255)
 
        left   = x
        right  = x + (2 * self._radius) + 1
        top    = y
        bottom = y + (2 * self._radius) + 1
        input_roi    = input_img[top:bottom, left:right]
        target_value = target_img[y, x]

        sample = {'image':input_roi, 'target':target_value}

        if self._transform:
            sample = self._transform(sample)

        return sample