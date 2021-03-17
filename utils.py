"""
Utility functions
"""

import numpy as np
import pdb

import scipy as sp
from mlpack import emst
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph

from hashlib import blake2b


def hash_param(param):
    """
    """
    s = ""
    for key, value in param.items():
        s += str(key) + "-" + str(value) + "--"
    h = blake2b()
    h.update(s.encode())
    return h.hexdigest()[:8]


class AverageMeter:
    def __init__(self, name='name'):
        self.name = name
        self.val = 0
        self.count = 0

    def update(self, val, count):
        self.val += val
        self.count += count

    @property
    def average(self):
        if self.count == 0:
            return 0
        else:
            return self.val/self.count


class EarlyStopping:
    def __init__(self, patience_max=10):
        self.patience_max = patience_max
        self.patience = patience_max
        self.current_score = 0.0

    def update(self, new_score):
        if new_score <= self.current_score:
            self.patience -= 1
        else:
            self.current_score = new_score
            self.patience = self.patience_max

    @property
    def stop(self):
        if self.patience <= 0:
            return True
        return False
