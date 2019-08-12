import os
import torch
import models
from definitions import MODELS_PATH


def get_model(kind, cache=False):
    if kind == 'sw':
        return models.SlideNet()
    elif kind == 'sk':
        return models.StridedNet()
    else:
        raise ValueError(f'Invalid kind value: \'kind\'')