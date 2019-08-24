import os
import config
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from definitions import MODELS_PATH


def get_weights(kind):
    """
    Returns the weights used in WeightedRandomSampler
    """
    cfg = config.read_config()
    data_path = cfg['data_path']
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