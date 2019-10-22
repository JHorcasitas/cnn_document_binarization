import os
import configparser

import cv2
import torch
import numpy as np
from tqdm import tqdm

import config
from constants import CONFIG_PATH


class Evaluator:
    def __init__(self, model, input_path, output_path=None):
        self._model = model
        self._input_path = input_path
        self._output_path = output_path
    
        self._config = config.read_config()
        self._model.eval()

    def evaluate(self):
        img = cv2.imread(self._input_path, 0)
        pad = np.pad(array=img,
                     pad_width=self._config['radius'], 
                     mode='constant', 
                     constant_values=255)
        out = np.empty_like(img).astype(np.float32)

        rows, cols = img.shape
        for row in tqdm(range(rows)):
            for col in range(cols):
                window = pad[row:(row + 2 * self._config['radius'] + 1), 
                             col:(col + 2 * self._config['radius'] + 1)]
                window = window /255
                window = (window - 0.733) / 0.129
                window = window[None, None, ...]
                window = torch.from_numpy(window).float()
                out[row, col] = torch.round(torch.sigmoid(self._model(window)))
                out[row, col] = out[row, col] * 255

        # save results
        if self._output_path:
            cv2.imwrite(os.path.join(self._output_path, 'out.jpg'), out)