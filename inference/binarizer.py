import configparser

import torch
import numpy as np
from tqdm import tqdm
from PIL import ImageOps
from torchvision import transforms
from torchvision.transforms import Compose

from models.model_factory import get_model


config = configparser.ConfigParser()
config.read('config.ini')


class Binarizer:
    def __init__(self, kind, device=None):
        """
        args:
            kind (str): one of {'sk', 'sw'}
            device (torch.device) defice to use in inference
        """
        self._kind = kind
        self._transform = Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.732],
                                                        std=[0.129])])

        if device:
            self._device = device
        else:
            self._device = torch.device('cpu')

        self._model = get_model(kind=kind,
                                device=self._device,
                                cache=True)
        self._model.eval()

        self._radius = config['DATA INGESTION'].getint('radius')

    def binarize(self, img):
        """
        args:
            img (PIL Image): image to binarize
        returns:
            binarized image
        """
        img = self._process_image(img)
        if self._kind == 'sw':
            return self._sw_binarization(img)
        elif self._kind == 'sk':
            pass
        else:
            raise ValueError(f'Unrecognized kind: "{self._kind}"')

    def _process_image(self, img):
        """
        Get img ready to be processed by the model
        """
        img = ImageOps.expand(img, border=self._radius, fill=255)
        img = self._transform(img)
        img = img[None, ...]
        return img

    def _sw_binarization(self, img):
        """
        """
        with torch.no_grad():
            rows = img.shape[2] - self._radius * 2
            cols = img.shape[3] - self._radius * 2
            output = np.empty((rows, cols)).astype(np.uint8)
            for row in tqdm(range(rows)):
                for col in range(cols):
                    window = img[0,
                                 0,
                                 row:(row + 2 * self._radius + 1),
                                 col:(col + 2 * self._radius + 1)]
                    window = window[None, None, ...]
                    win_out = self._model(window)
                    win_out = torch.round(torch.sigmoid(win_out)) * 255
                    win_out = int(win_out.item())
                    output[row, col] = win_out
        return output
