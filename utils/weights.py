import os
import configparser
from collections import Counter

import numpy as np
from PIL import Image


config = configparser.ConfigParser()
config.read('config.ini')


def get_weights(kind):
    """
    Returns the weights used in WeightedRandomSampler
    """
    data_path = config['data_ingestion']['data_path']
    input_path = os.path.join(data_path, 'target', kind)

    distribution = []
    order = lambda x: int(os.path.splitext(x)[0])
    for img_name in sorted(os.listdir(input_path), key=order):
        img_path = os.path.join(input_path, img_name)
        img = Image.open(img_path).convert('L')
        img = np.array(img)
        img = img.ravel()
        img = np.where(img > 0, 1, 0)
        distribution.extend(img.tolist())

    counts = Counter(distribution)
    weights = [1 / counts[item] for item in distribution]
    return weights
