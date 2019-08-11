import os
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

import config


class BinaryDataset(Dataset):
    """
    Retrieve image patches
    """
    def __init__(self, kind, input_transform=None):
        """
        args:
        kind (str): one of: {train, test, val}
        transform (): transformation to apply to every image
        """
        cfg = config.read_config()

        self._kind       = kind
        self._radius     = cfg['radius']
        self._data_path  = cfg['data_path']
        self._input_transform  = input_transform
        
        input_path  = os.path.join(self._data_path, 'input', self._kind)
        target_path = os.path.join(self._data_path, 'target', self._kind)
        self._input_imgs  = self._load_images(input_path, 'L')
        self._target_imgs = self._load_images(target_path, '1')

        self._img_pixels = self._get_num_pixels()
    
    def _load_images(self, path, mode):
        img_dict = OrderedDict()
        order = lambda x: int(os.path.splitext(x)[0])
        for img_name in sorted(os.listdir(path), key=order):
            img_path = os.path.join(path, img_name)
            img_dict[img_name] = Image.open(img_path).convert(mode)
        return img_dict

    def _get_num_pixels(self):
        """
        Get the number of pixels for every image in 'self._input_imgs'
        """
        num_pixels = OrderedDict()
        for key in self._input_imgs:
            width, height = self._input_imgs[key].size
            num_pixels[key] = width * height
        return num_pixels

    def __len__(self):
        """
        Every pixel represents a training pattern
        """
        num_patterns = 0
        for key in self._img_pixels:
            num_patterns += self._img_pixels[key]
        return num_patterns
    
    def __getitem__(self, idx):

        def get_img_info(idx):
            total = 0
            for key in self._img_pixels:
                total += self._img_pixels[key]
                if idx < total:
                    return key, idx - (total - self._img_pixels[key]) + 1
            raise RuntimeError(f'Index {idx} could not be found')
        
        def get_coord_from_idx(input_img, img_idx):
            w, h = input_img.size
            row = (img_idx - 1) // w
            col = (img_idx - 1) % w
            return col, row

        img_name, img_index = get_img_info(idx)
        input_img  = self._input_imgs[img_name] 
        input_img = ImageOps.expand(input_img, border=self._radius, fill=255)
        target_img = self._target_imgs[img_name]
        
        x, y = get_coord_from_idx(input_img, img_index)

        # Get input
        right  = x + (2 * self._radius) + 1
        bottom = y + (2 * self._radius) + 1
        input = input_img.crop((x, y, right, bottom))

        # Get target
        target = target_img.getpixel((x, y))
        target = 1 if target > 0 else 0
        target = np.array(target, dtype=np.float32).reshape(1)

        if self._input_transform:
            input = self._input_transform(input)

        return input, target


class StridedDataset(Dataset):
    """
    Retrieve complete images, useful when training fully convolutional
    networks 
    """
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

    def __len__(self):
        return len(self._input_names)
    
    def __getitem__(self, idx):
        input_path = os.path.join(self._input_path, self._input_names[idx])
        input = Image.open(input_path).convert('L')

        target_path = os.path.join(self._target_path, self._target_names[idx])
        target = Image.open(target_path).convert('1')

        if self._input_transform:
            input = self._input_transform(input)
        if self._target_transform:
            target = self._target_transform(target)
    
        return input, target