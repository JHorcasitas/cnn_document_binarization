import os
import configparser
from typing import List
from collections import Counter

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


config = configparser.ConfigParser()
config.read('config.ini')


def get_weights(dataset: Dataset) -> List[float]:
    """Returns the weights used in WeightedRandomSampler"""
    data_path = config['data_ingestion']['data_path']
    input_path = os.path.join(data_path, 'target', dataset._kind)

    distribution = []
    for img_name in sorted(os.listdir(input_path), key=basename_order):
        img_path = os.path.join(input_path, img_name)
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        img = img.ravel()
        img = np.where(img > 0, 1, 0)
        distribution.extend(img.tolist())

    counts = Counter(distribution)
    weights = [1 / counts[item] for item in distribution]
    return weights


def basename_order(img_name: str) -> int:
    return int(os.path.splitext(img_name)[0])
