import os
import configparser
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .data_transformations import input_transform, target_transform


config = configparser.ConfigParser()
config.read('config.ini')


class BinaryDataset(Dataset):
    """
    Retrieve image patches
    args:
        kind (str): one of: {train, test, val}
        input_transform (callable): transformation to apply to every image
    """
    def __init__(self, kind, transform_input=True, transform_target=True):
        self._kind = kind
        self._radius = config['data_ingestion'].getint('radius')
        self._data_path = config['data_ingestion']['data_path']
        self._transform_input = transform_input
        self._transform_target = transform_target

        input_path = os.path.join(self._data_path, 'input', self._kind)
        target_path = os.path.join(self._data_path, 'target', self._kind)
        self._input_imgs = self._load_images(input_path, mode='L')
        self._target_imgs = self._load_images(target_path, mode='1')

        self._img_pixels = self._get_num_pixels()

    def _load_images(self, path, mode):
        img_dict = OrderedDict()
        image_list = sorted(os.listdir(path),
                            key=lambda x: int(os.path.splitext(x)[0]))
        for img_name in image_list:
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
        """ Every pixel (with its surrounding area) is a training pattern """
        return sum(self._img_pixels.values())

    def __getitem__(self, idx):

        def get_img_name_and_index(idx):
            total = 0
            for key in self._img_pixels:
                total += self._img_pixels[key]
                if idx < total:
                    return key, idx - (total - self._img_pixels[key]) + 1

        def get_coord_from_idx(input_img, img_idx):
            w, h = input_img.size
            row = (img_idx - 1) // w
            col = (img_idx - 1) % w
            return col, row

        img_name, img_index = get_img_name_and_index(idx)
        input_img = self._input_imgs[img_name]
        target_img = self._target_imgs[img_name]

        x, y = get_coord_from_idx(input_img, img_index)
        input_img = ImageOps.expand(input_img, border=self._radius, fill=255)

        # Get input
        right = x + (2 * self._radius) + 1
        bottom = y + (2 * self._radius) + 1
        input = input_img.crop((x, y, right, bottom))

        # Get target
        target = target_img.getpixel((x, y))
        target = 1 if target > 0 else 0
        target = np.array(target, dtype=np.float32).reshape(1)

        if self._transform_input:
            input = input_transform(input)

        if self._transform_target:
            target = torch.from_numpy(target)

        return input, target


class StridedDataset(Dataset):
    """
    Retrieve complete images, useful when training fully convolutional
    networks
    Args:
        kind (str): one of: {train, test, val}
        transform_input (bool): wether to transform input so that it can be
        processed by PyTorch
        transform_target (bool): wether to transform target so that it can be
        processed by Pytorch
    """
    def __init__(self, kind, transform_input=False, transform_target=False):
        self._kind = kind
        self._transform_input = transform_input
        self._transform_target = transform_target

        self._radius = config['data_ingestion'].getint('radius')
        self._data_path = config['data_ingestion']['data_path']

        self._input_path = os.path.join(self._data_path, 'input', self._kind)
        self._target_path = os.path.join(self._data_path, 'target', self._kind)

        order = lambda x: int(os.path.splitext(x)[0])
        self._input_names = sorted(os.listdir(self._input_path), key=order)
        self._target_names = sorted(os.listdir(self._target_path), key=order)

    def __len__(self):
        return len(self._input_names)

    def __getitem__(self, idx):
        input_path = os.path.join(self._input_path, self._input_names[idx])
        input = Image.open(input_path).convert('L')

        target_path = os.path.join(self._target_path, self._target_names[idx])
        target = Image.open(target_path).convert('1')

        if self._transform_input:
            input = input_transform(input)
        if self._transform_target:
            target = target_transform(target)

        return input, target
