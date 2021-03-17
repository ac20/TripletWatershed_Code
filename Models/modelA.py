"""
ResNet Architecture for computing the representations.
"""
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class modelA(nn.Module):
    def __init__(self, **param):
        super().__init__()
        self.number_features = param['number_features']
        self.embed_dim = param['embed_dim']
        self.number_labels = param['number_labels']
        self.patch_size = 5

        self.bn1 = nn.BatchNorm2d(self.number_features)
        self.conv1 = nn.Conv2d(self.number_features, 64, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, bias=False)
        sz = 2*self.patch_size + 1 - 2*(3-1)

        self.bn4 = nn.BatchNorm1d(16*sz*sz)
        self.fc = nn.Linear(16*sz*sz, self.embed_dim)

    def forward_rep(self, x):
        out = F.relu(self.conv1(self.bn1(x)))
        out = F.relu(self.conv2(self.bn2(out)))
        out = F.relu(self.conv3(self.bn3(out)))
        out = torch.reshape(out, (len(out), -1))
        out = self.fc(self.bn4(out))
        return out

    def forward(self, x):
        out = self.forward_rep(x)
        return out
