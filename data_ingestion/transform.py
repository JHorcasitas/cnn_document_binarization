import torch


class Normalize:
    """
    Normalizes by substracting mean and dividing by standard deviation.
    Normalization is expected to happen before Tensorization
    """
    def __init__(self, mean, std):
        self._std  = std
        self._mean = mean

    def __call__(self, sample):
        img = sample['image']
        img = (img - self._mean) / self._std
        sample['image'] = img
        return sample

class Tensorization:
    """
    Convert to PyTorch Tensor and adds dummy dimension
    """
    def __call__(self, sample):
        img = sample['image']
        img = img[None, ...]
        sample['image']  = torch.from_numpy(img).float()

        target = torch.tensor(sample['target'])
        sample['target'] = target.unsqueeze(0)
        return sample