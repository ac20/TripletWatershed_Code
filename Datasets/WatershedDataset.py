

import numpy as np
import pdb

import torch
from torch.utils.data import Dataset, DataLoader


class watershed_augment_dataset(Dataset):
    """Augment the dataset to use watershed labels.
    """

    def __init__(self, base_dataset, **param):
        self.base_dataset = base_dataset
        self.watershed_labels = np.zeros(len(base_dataset), dtype=np.int64)
        self.number_labels = None

        self.train_idx, self.test_idx = self.base_dataset.train_idx, self.base_dataset.test_idx

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X, _ = self.base_dataset.__getitem__(idx)
        y = self.watershed_labels[idx]
        return X, y

    def update_watershed_labels(self, new_labels):
        self.watershed_labels = new_labels
        self.number_labels = np.max(self.watershed_labels)

    @property
    def labels(self):
        return self.watershed_labels
