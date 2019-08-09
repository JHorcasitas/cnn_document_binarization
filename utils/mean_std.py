import os
from statistics import mean

from torchvision import transforms
from torchvision.transforms import Compose

import data_ingestion
from definitions import UTILS_PATH


def main():
    """
    Computes and writes the mean and std of the training data
    """
    # Load Datasets
    input_transform  = Compose([transforms.ToTensor()])
    dataset_factory = data_ingestion.DatasetFactory()
    dataset = dataset_factory.get_dataset(kind='train',
                                          input_transform=input_transform)

    data_mean, data_std = [], []
    for input, _ in dataset:
        data_mean.append(input.mean().item())
        data_std.append(input.std().item())

    write_file = os.path.join(UTILS_PATH, 'std_mean.txt')
    with open(write_file, 'w') as f:
        f.write(f'Mean: {mean(data_mean)}\n')
        f.write(f'Standard Deviation: {mean(data_std)}')

if __name__ == '__main__':
    main()