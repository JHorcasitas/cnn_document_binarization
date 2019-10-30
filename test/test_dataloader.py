import unittest
from random import random
from functools import reduce

from data_ingestion.dataset import BinaryDataset
from data_ingestion.dataloader import BinaryDataLoader, MAX_DATASET_SIZE


class TestDataloader(unittest.TestCase):

    def setUp(self):
        self.dataset = dataset = BinaryDataset(kind='train')
        self.weights = [random() for _ in range(len(dataset))]
        self.loader = BinaryDataLoader(dataset=self.dataset,
                                       weights=self.weights)

    def test_split_dataset(self):
        # Check that the length of the original dataset is bigger than the
        # allowed by PyTorch 1.3.0
        self.assertTrue(len(self.dataset) > MAX_DATASET_SIZE)

        # The sum of the length of each dataset must be equal to the the length
        # of the original dataset
        cum_dataset_length = sum([len(d) for d in self.loader._datasets])
        self.assertEqual(len(self.dataset), cum_dataset_length)

        # Check that the length of each dataset is smaller than what PyTorch
        # allows
        for d in self.loader._datasets:
            self.assertTrue(len(d) <= MAX_DATASET_SIZE)

        # Check that indices of all datasets are different
        indices = [d.indices for d in self.loader._datasets]
        concat_indices = reduce(lambda acc, v: acc + v, indices, initial=[])
        self.assertEqual(len(self.dataset), len(set(concat_indices)))

        # Check that no index is bigger than the original dataset length
        self.assertTrue(max(concat_indices) == len(self.dataset))

    def test_split_weights(self):
        pass

    def test_construct_dataloaders(self):
        pass
