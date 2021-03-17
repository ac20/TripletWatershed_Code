"""
This module defines the method to obtain the watershed labels for the given dataset.
"""


import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

import higra as hg
from mlpack import emst
import scipy as sp


def _euclid_distance(X, u, v):
    """
    """
    return np.sqrt(np.sum((X[u, :]-X[v, :])**2, axis=1))


def _split_seeds_train(indexes, labels, percent_seeds):
    """Split the index set into seeds and train
    """
    indseed = []
    indtrain = []
    for l in range(np.max(labels)):
        arr = indexes[labels[indexes] == l+1]
        np.random.shuffle(arr)
        cutpoint = max(int(percent_seeds*len(arr)), 1) + 1
        indseed += list(arr[:cutpoint])
        indtrain += list(arr[cutpoint:])
    return np.array(indseed, dtype=np.int32), np.array(indtrain, dtype=np.int32)


def _get_rep(model, traindata, **param):
    """
    """
    Xrep = []
    labels = []
    dataloader = DataLoader(traindata, batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch_no, (X, y) in enumerate(dataloader):
            X = X.to(param['device'])
            tmp = model.forward_rep(X)
            Xrep.append(tmp.cpu().detach().numpy())
            labels.append(y.cpu().detach().numpy())

    Xrep = np.concatenate(Xrep, axis=0)
    labels = np.concatenate(labels, axis=0)
    return Xrep, labels


def _induced_edges(traindata, indexes):
    utmp, vtmp = traindata.edges
    arr = np.zeros(len(traindata))
    arr[indexes] = 1
    indselect = np.logical_and(arr[utmp] == 1, arr[vtmp] == 1)
    utmp, vtmp = utmp[indselect], vtmp[indselect]

    arr2 = -1*np.ones(len(traindata))
    arr2[indexes] = np.arange(len(indexes))
    uedge = np.array(arr2[utmp], dtype=np.int64, copy=True)
    assert np.all(uedge >= 0)
    vedge = np.array(arr2[vtmp], dtype=np.int64, copy=True)
    assert np.all(vedge >= 0)
    return uedge, vedge


def get_watershed_labels(model, traindata, **param):
    """Return the watershed labels.

    Note: Watershed is computed only on the induced subgraph of {train_idx + test_idx}
    """

    ind_original = np.concatenate((traindata.train_idx, traindata.test_idx))
    subset = Subset(traindata, ind_original)
    Xrep, labels = _get_rep(model, subset, **param)
    size_data, number_features = np.shape(Xrep)
    indtrain = np.arange(len(traindata.train_idx))
    indseed, indvalid = _split_seeds_train(indtrain, labels, percent_seeds=0.4)

    size_select = min(int(0.4*number_features), 32)
    features_choose = np.random.choice(np.arange(number_features), size=size_select, replace=False)
    uedge, vedge = _induced_edges(traindata, ind_original)
    wedge = _euclid_distance(Xrep[:, features_choose], uedge, vedge) + 1e-6

    graph = hg.UndirectedGraph()  # pylint: disable=no-member
    graph.add_vertices(size_data)
    graph.add_edges(uedge, vedge)

    vertex_seeds = np.zeros(size_data, dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)
    vertex_seeds[indseed] = labels[indseed]
    watershed_labels = hg.labelisation_seeded_watershed(graph, wedge, vertex_seeds, background_label=0)

    final_labels = np.zeros(len(traindata), dtype=np.int32)
    final_labels[ind_original] = watershed_labels

    if len(indvalid) > 0:
        watershed_acc = np.mean(watershed_labels[indvalid] == labels[indvalid])
        # print("Watershed Accuracy (Train): {:0.4f}".format(watershed_acc))
    else:
        watershed_acc = 0.0
    return final_labels, watershed_acc
