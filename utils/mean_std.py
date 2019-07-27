"""
Computes the mean and std for the training data
Run with the -m flag
"""
import os
import cv2
from statistics import mean
from cnn_document_binarization import config
from cnn_document_binarization.constants import UTILS_PATH


def main():
    cfg = config.read_config()
    data_path = cfg['data_path']
    data_train_path = os.path.join(data_path, 'input', 'train')

    data_mean, data_std = [], []
    for file in os.scandir(data_train_path):
        img = cv2.imread(file.path, 0) / 255
        data_mean.append(img.mean())
        data_std.append(img.std())

    write_file = os.path.join(UTILS_PATH, 'std_mean.txt')
    with open(write_file, 'w') as f:
        f.write(f'Mean: {mean(data_mean)}\n')
        f.write(f'Standard Deviation: {mean(data_std)}')

if __name__ == '__main__':
    main()