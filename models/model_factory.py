import os
import torch
from definitions import MODELS_PATH
from .models.sk_network import StridedNet


def get_model(cache=False):
    model = StridedNet()
    if cache:
        pass
    return model