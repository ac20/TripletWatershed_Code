"""
Indian Pines Dataset
"""

import numpy as np
import pdb
import os
import wget

import scipy as sp
from scipy.io import loadmat

from sklearn.decomposition import PCA
from sklearn import metrics, preprocessing
from mlpack import emst

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms


def get_2d_index(idx, s0, s1):
    ida, idb = idx//s1, idx % s1
    if ida >= s0:
        raise Exception("index exceeds dimensions")
    return ida, idb


def get_emst_edges(img, labels):
    """EMST edges on the {labels!=0} vertices
    """
    s0, s1, _ = np.shape(img)
    X = img.reshape((s0*s1, -1))
    Xpca = PCA(n_components=32).fit_transform(X)
    indselect = np.where(labels != 0)[0]
    mst = emst(Xpca[indselect])['output']
    uedge = np.array(mst[:, 0], dtype=np.int64)
    vedge = np.array(mst[:, 1], dtype=np.int64)
    uedge, vedge = indselect[uedge], indselect[vedge]
    return uedge, vedge


def get_4adj_edges(img, labels):
    """
    """
    s0, s1 = img.shape[:2]
    number_pixels = s0*s1
    z = np.arange(s0*s1).reshape((s0, s1))
    uedge = np.concatenate((z[:-1, :].flatten(), z[:, :-1].flatten()))
    vedge = np.concatenate((z[1:, :].flatten(), z[:, 1:].flatten()))
    assert len(uedge) == len(vedge)
    assert len(uedge) == (s0-1)*s1 + (s1-1)*s0
    return uedge, vedge


def get_edges_4adjEMST(img, labels):
    """These edges are constructed by ignoring {labels==0} points and adding the Euclidean
    Minimum Spanning Tree of all the {labels!=0} points.

    img has shape (s0,s1,s2)
    labels has the shape (s0*s1,)

    """
    # Get the 4adj edges
    u1, v1 = get_4adj_edges(img, labels)

    # Add EMST edges for all points with label != 0
    u2, v2 = get_emst_edges(img, labels)

    # Combine both EMST and 4adj edges
    uedge = np.concatenate((u1, u2))
    vedge = np.concatenate((v1, v2))

    return uedge, vedge


def train_test_split_random(labels, **param):
    seed = param['seed']
    train_size = param['train_size']
    np.random.seed(seed)
    train = []
    test = []
    for l in range(np.max(labels)):
        arr = np.where(labels == l+1)[0]
        np.random.shuffle(arr)
        cutpoint = max(int((train_size) * len(arr)), 4)
        train += list(arr[:cutpoint])
        test += list(arr[cutpoint:])
    return np.array(train, dtype=np.int32), np.array(test, dtype=np.int32)


def train_test_split_semisupervised(labels, **param):
    seed = param['seed']
    train_size = param['train_size']
    np.random.seed(seed)
    train = []
    test = []
    for l in range(np.max(labels)):
        arr = np.where(labels == l+1)[0]
        np.random.shuffle(arr)
        if len(arr) > 30:
            cutpoint = 30
        else:
            cutpoint = 15
        train += list(arr[:cutpoint])
        test += list(arr[cutpoint:])
    return np.array(train, dtype=np.int32), np.array(test, dtype=np.int32)


def get_indianpines_data(root="./data", download=True):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"

    if (not os.path.exists(root+"/Indian_pines_corrected.mat")) and download:
        wget.download(data_url, root + "/Indian_pines_corrected.mat")

    if (not os.path.exists(root+"/Indian_pines_gt.mat")) and download:
        wget.download(gt_url, root + "/Indian_pines_gt.mat")

    data = loadmat("./data/Indian_pines_corrected.mat")
    X = np.array(data['indian_pines_corrected'], dtype=np.float32)

    # Normalize
    # muX = np.mean(X, axis=(0, 1), keepdims=True)
    # stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    # X = (X - muX)/stdX

    data = loadmat("./data/Indian_pines_gt.mat")
    y = np.array(data['indian_pines_gt'], dtype=np.int32)
    return X, y


def get_paviaU_data(root="./data", download=True):
    """
    """
    data_url = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
    gt_url = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"

    if (not os.path.exists(root+"/PaviaU.mat")) and download:
        wget.download(data_url, root + "/PaviaU.mat")

    if (not os.path.exists(root+"/PaviaU_gt.mat")) and download:
        wget.download(gt_url, root + "/PaviaU_gt.mat")

    data = loadmat("./data/PaviaU.mat")
    X = np.array(data['paviaU'], dtype=np.float32)

    # Normalize
    muX = np.mean(X, axis=(0, 1), keepdims=True)
    stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    X = (X - muX)/stdX

    data = loadmat("./data/PaviaU_gt.mat")
    y = np.array(data['paviaU_gt'], dtype=np.int32)
    return X, y


def get_ksc_data(root="./data", download=True):
    """
    """
    data_url = "http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat"
    gt_url = "http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat"

    if (not os.path.exists(root+"/KSC.mat")) and download:
        wget.download(data_url, root + "/KSC.mat")

    if (not os.path.exists(root+"/KSC_gt.mat")) and download:
        wget.download(gt_url, root + "/KSC_gt.mat")

    data = loadmat("./data/KSC.mat")
    X = np.array(data['KSC'], dtype=np.float32)

    # Normalize
    muX = np.mean(X, axis=(0, 1), keepdims=True)
    stdX = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
    X = (X - muX)/stdX

    data = loadmat("./data/KSC_gt.mat")
    y = np.array(data['KSC_gt'], dtype=np.int32)
    return X, y


class Hyperspectral_Dataset(Dataset):
    """
    Required parameters:
    param[''patch_size'] = {None, int}
    param['edges'] = {'4adj', 'connected'}
    param['dataset'] = {'indianpines', 'paviaU', 'ksc'}
    """

    def __init__(self, root="./data", download=True, **param):

        # Set the patch_size
        self.patch_size = 5

        # Get the data
        if param['dataset'] == 'indianpines':
            self.X, self.y = get_indianpines_data(root, download)
        if param['dataset'] == 'paviaU':
            self.X, self.y = get_paviaU_data(root, download)
        if param['dataset'] == 'ksc':
            self.X, self.y = get_ksc_data(root, download)

        self.uedge, self.vedge = get_edges_4adjEMST(self.X, self.y.flatten())

        # transform X,y using PCA
        self.sx, self.sy, self.sz = np.shape(self.X)
        self.X = self.X.reshape((-1, self.sz))
        self.X = PCA(n_components=self.sz).fit_transform(self.X)
        self.X = self.X.reshape((self.sx, self.sy, self.sz))
        muX = np.mean(self.X, axis=(0, 1), keepdims=True)
        stdX = np.std(self.X, axis=(0, 1), keepdims=True) + 1e-6
        self.X = (self.X - muX)/stdX

        # Split the data into train and test
        self.train_idx, self.test_idx = train_test_split_random(self.y.flatten(), **param)

        if 'semi_supervised' in param.keys():
            self.train_idx, self.test_idx = train_test_split_semisupervised(self.y.flatten(), **param)

    def __len__(self):
        return len(self.y.flatten())

    def __getitem__(self, idx):
        return self._getitem_patch(idx)

    def _getitem_patch(self, idx):
        idx_select = idx
        idxa, idxb = get_2d_index(idx_select, self.sx, self.sy)
        lowerx, pad_lowerx = max(idxa-self.patch_size, 0), -1*min(idxa-self.patch_size, 0)
        upperx, pad_upperx = min(idxa+self.patch_size+1, self.sx), max(idxa+self.patch_size+1-self.sx, 0)
        lowery, pad_lowery = max(idxb-self.patch_size, 0), -1*min(idxb-self.patch_size, 0)
        uppery, pad_uppery = min(idxb+self.patch_size+1, self.sy), max(idxb+self.patch_size+1-self.sy, 0)
        img = torch.from_numpy(self.X[lowerx:upperx, lowery:uppery, :]).permute(2, 0, 1)
        img = F.pad(img, (pad_uppery, pad_lowery, pad_lowerx, pad_upperx), mode='constant', value=0)
        label = self.y[idxa, idxb]
        return img, label

    @property
    def number_features(self):
        return np.shape(self.X)[-1]

    @property
    def edges(self):
        return self.uedge, self.vedge

    @property
    def number_labels(self):
        return np.max(self.y)

    @property
    def labels(self):
        return self.y.flatten()
