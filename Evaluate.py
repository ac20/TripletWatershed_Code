"""
Evaluation Module. We estimate three metrics - OA, AA, and kappa using the watershed layer.
"""

import numpy as np
import pdb
import pickle
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import higra as hg

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN


def _euclid_distance(X, u, v):
    dist = np.sum((X[u, :] - X[v, :])**2, axis=1) + 1e-6*np.random.rand(len(u))
    return np.array(dist, dtype=np.float64)


def _split_seeds_train(indexes, labels, percent_seeds):
    """Split the index set into seeds and train
    """
    indseed = []
    indtrain = []
    for l in range(np.max(labels)):
        arr = indexes[labels[indexes] == l+1]
        np.random.shuffle(arr)
        cutpoint = max(int(percent_seeds*len(arr)), 1)
        indseed += list(arr[:cutpoint])
        indtrain += list(arr[cutpoint:])
    return np.array(indseed, dtype=np.int32), np.array(indtrain, dtype=np.int32)


def _get_pass_value(bpt, lca_fast, indseed, indtrain, wedge):
    """
    """
    wtemp = np.concatenate((wedge, np.array([1e-6])))
    inda, indb = np.meshgrid(indtrain, indseed)
    sx, sy = np.shape(inda)
    inda, indb = inda.flatten(), indb.flatten()
    lca_list = lca_fast.lca(inda, indb) - bpt.num_leaves()
    lca_list[inda == indb] = 0
    mst_edge_map = hg.get_attribute(bpt, 'mst_edge_map')
    lca_list = mst_edge_map[lca_list]
    lca_list[inda == indb] = len(wedge)
    lca_list = lca_list.reshape((sx, sy)).transpose()
    passvalue = wtemp[lca_list]
    return np.min(passvalue, axis=1)


def _compute_OA(prob, labels, indtrain, indtest):
    """
    """
    predlabels = np.argmax(prob, axis=1)+1
    acc_train = np.mean(predlabels[indtrain] == labels[indtrain])
    acc_test = np.mean(predlabels[indtest] == labels[indtest])
    return acc_train, acc_test


def _compute_AA(prob, labels, indtrain, indtest):
    """
    """
    predlabels = np.argmax(prob, axis=1)+1
    acc_train = 0
    acc_test = 0
    count = 0
    for l in range(np.max(labels)):
        indselect = indtrain[labels[indtrain] == l+1]
        acc_train += np.mean(predlabels[indselect] == labels[indselect])
        indselect = indtest[labels[indtest] == l+1]
        acc_test += np.mean(predlabels[indselect] == labels[indselect])
        count += 1
    return acc_train/count, acc_test/count


def _compute_accuracy_classwise(prob, labels, indtrain, indtest):
    """
    """
    predlabels = np.argmax(prob, axis=1)+1
    acc_train = []
    acc_test = []
    count = 0
    for l in range(np.max(labels)):
        indselect = indtrain[labels[indtrain] == l+1]
        acc_train.append(np.mean(predlabels[indselect] == labels[indselect]))
        indselect = indtest[labels[indtest] == l+1]
        acc_test.append(np.mean(predlabels[indselect] == labels[indselect]))
        count += 1
    return acc_train, acc_test


def _compute_kappa(prob, labels, indtrain, indtest):
    """
    """
    predlabels = np.argmax(prob, axis=1)+1
    kappa_train = cohen_kappa_score(predlabels[indtrain], labels[indtrain])
    kappa_test = cohen_kappa_score(predlabels[indtest], labels[indtest])
    return kappa_train, kappa_test


def _compute_rep(model, traindata, **param):
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
    labels = np.concatenate(labels)
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


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == 19:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def _plot_image(traindata, prob, fname):
    """
    """
    predlabels = np.argmax(prob, axis=1)+1
    indtrain, indtest = traindata.train_idx, traindata.test_idx
    idx_indtrain = np.arange(len(traindata.train_idx))
    idx_indtest = len(indtrain) + np.arange(len(traindata.test_idx))

    sx, sy, sz = traindata.sx, traindata.sy, traindata.sz
    img = np.zeros((sx, sy), dtype=np.int32)
    img = img.flatten()
    img[indtrain] = predlabels[idx_indtrain]
    img[indtest] = predlabels[idx_indtest]
    img = img.reshape((sx, sy))

    cmap_arr = np.array(list_to_colormap(np.arange(20)))
    img[img == 0] = 17
    img = img-1

    plt.imsave(fname, cmap_arr[img])


def evaluate_model_watershed(model, traindata, fname=None, return_labels=False, **param):
    """
    """
    ind_original = np.concatenate((traindata.train_idx, traindata.test_idx))
    indtrain = np.arange(len(traindata.train_idx))
    indtest = len(indtrain) + np.arange(len(traindata.test_idx))

    subset = Subset(traindata, ind_original)
    Xrep, labels = _compute_rep(model, subset, **param)

    number_labels = np.max(labels)
    size_data, number_features = np.shape(Xrep)

    prob = np.zeros((len(Xrep), number_labels))
    weight = 0
    for rep in range(20):

        size_select = min(int(0.4*number_features), 32)
        feature_select = np.random.choice(np.arange(number_features), size=size_select, replace=False)

        uedge, vedge = _induced_edges(traindata, ind_original)
        wedge = _euclid_distance(Xrep[:, feature_select], uedge, vedge)

        graph = hg.UndirectedGraph()  # pylint: disable=no-member
        graph.add_vertices(len(Xrep))
        graph.add_edges(uedge, vedge)
        bpt, alt = hg.bpt_canonical(graph, wedge)
        lca_fast = hg.make_lca_fast(bpt)

        indseed, indvalid = _split_seeds_train(indtrain, labels, percent_seeds=0.5)
        probtmp = np.zeros((len(Xrep), number_labels))
        for l in range(number_labels):
            indseedSelect = indseed[labels[indseed] == l+1]
            probtmp[:, l] = _get_pass_value(bpt, lca_fast, indseedSelect, np.arange(len(Xrep)), wedge)
        probtmp = np.exp(-1*probtmp/probtmp.std())
        probtmp = probtmp/np.sum(probtmp, axis=1, keepdims=True)

        predvalid = np.argmax(probtmp[indvalid, :], axis=1) + 1
        weighttmp = np.mean(predvalid == labels[indvalid])

        prob += probtmp*weighttmp
        weight += weighttmp
        # print("\r{} out of {} done...".format(rep+1, 20), end="")
    prob = prob/weight

    if fname is not None:
        _plot_image(traindata, prob, fname)

    if return_labels:
        predlabels = np.argmax(prob, axis=1)+1
        return predlabels

    OA_train, OA_test = _compute_OA(prob, labels, indtrain, indtest)
    AA_train, AA_test = _compute_AA(prob, labels, indtrain, indtest)
    kappa_train, kappa_test = _compute_kappa(prob, labels, indtrain, indtest)
    return OA_train, AA_train, kappa_train, OA_test, AA_test, kappa_test


def evaluate_model_watershed_classwise(model, traindata, valid=True, **param):
    """
    """
    ind_original = np.concatenate((traindata.train_idx, traindata.test_idx))
    indtrain = np.arange(len(traindata.train_idx))
    indtest = len(indtrain) + np.arange(len(traindata.test_idx))

    subset = Subset(traindata, ind_original)
    Xrep, labels = _compute_rep(model, subset, **param)

    number_labels = np.max(labels)
    size_data, number_features = np.shape(Xrep)

    prob = np.zeros((len(Xrep), number_labels))
    weight = 0
    for rep in range(20):

        size_select = min(int(0.4*number_features), 32)
        feature_select = np.random.choice(np.arange(number_features), size=size_select, replace=False)

        uedge, vedge = _induced_edges(traindata, ind_original)
        wedge = _euclid_distance(Xrep[:, feature_select], uedge, vedge)

        graph = hg.UndirectedGraph()  # pylint: disable=no-member
        graph.add_vertices(len(Xrep))
        graph.add_edges(uedge, vedge)
        bpt, alt = hg.bpt_canonical(graph, wedge)
        lca_fast = hg.make_lca_fast(bpt)

        indseed, indvalid = _split_seeds_train(indtrain, labels, percent_seeds=0.5)
        probtmp = np.zeros((len(Xrep), number_labels))
        for l in range(number_labels):
            indseedSelect = indseed[labels[indseed] == l+1]
            probtmp[:, l] = _get_pass_value(bpt, lca_fast, indseedSelect, np.arange(len(Xrep)), wedge)
        probtmp = np.exp(-1*probtmp/probtmp.std())
        probtmp = probtmp/np.sum(probtmp, axis=1, keepdims=True)

        predvalid = np.argmax(probtmp[indvalid, :], axis=1) + 1
        weighttmp = np.mean(predvalid == labels[indvalid])

        prob += probtmp*weighttmp
        weight += weighttmp
        # print("\r{} out of {} done...".format(rep+1, 20), end="")
    prob = prob/weight

    acc_train, acc_test = _compute_accuracy_classwise(prob, labels, indtrain, indtest)
    OA_train, OA_test = _compute_OA(prob, labels, indtrain, indtest)
    AA_train, AA_test = _compute_AA(prob, labels, indtrain, indtest)
    kappa_train, kappa_test = _compute_kappa(prob, labels, indtrain, indtest)
    return OA_train, AA_train, kappa_train, acc_train, OA_test, AA_test, kappa_test, acc_test


def evaluate_ensemble_watershed(traindata, Xrep, labels, train_indices, test_indices):
    """
    """
    ind_original = np.concatenate((train_indices, test_indices))
    indtrain = np.arange(len(traindata.train_idx))
    indtest = len(indtrain) + np.arange(len(traindata.test_idx))

    number_labels = int(np.max(labels))
    size_data, number_features = np.shape(Xrep)

    prob = np.zeros((len(Xrep), number_labels))
    weight = 0
    for rep in range(20):

        size_select = 64
        feature_select = np.random.choice(np.arange(number_features), size=size_select, replace=False)

        uedge, vedge = _induced_edges(traindata, ind_original)
        wedge = _euclid_distance(Xrep[:, feature_select], uedge, vedge)
        graph = hg.UndirectedGraph()  # pylint: disable=no-member
        graph.add_vertices(len(Xrep))
        graph.add_edges(uedge, vedge)
        bpt, alt = hg.bpt_canonical(graph, wedge)
        lca_fast = hg.make_lca_fast(bpt)

        indseed, indvalid = _split_seeds_train(indtrain, labels, percent_seeds=0.5)
        probtmp = np.zeros((len(Xrep), number_labels))
        for l in range(number_labels):
            indseedSelect = indseed[labels[indseed] == l+1]
            probtmp[:, l] = _get_pass_value(bpt, lca_fast, indseedSelect, np.arange(len(Xrep)), wedge)
        probtmp = np.exp(-1*probtmp/probtmp.std())
        probtmp = probtmp/np.sum(probtmp, axis=1, keepdims=True)

        predvalid = np.argmax(probtmp[indvalid, :], axis=1) + 1
        weighttmp = np.mean(predvalid == labels[indvalid])

        prob += probtmp*weighttmp
        weight += weighttmp
        print("\r{} out of {} done...".format(rep+1, 20), end="")
    prob = prob/weight

    acc_train, acc_test = _compute_accuracy_classwise(prob, labels, indtrain, indtest)
    OA_train, OA_test = _compute_OA(prob, labels, indtrain, indtest)
    AA_train, AA_test = _compute_AA(prob, labels, indtrain, indtest)
    kappa_train, kappa_test = _compute_kappa(prob, labels, indtrain, indtest)
    return OA_train, AA_train, kappa_train, acc_train, OA_test, AA_test, kappa_test, acc_test


def evaluate_rf_clf(model, traindata, **param):
    """
    """
    ind_original = np.concatenate((traindata.train_idx, traindata.test_idx))
    indtrain = np.arange(len(traindata.train_idx))
    indtest = len(indtrain) + np.arange(len(traindata.test_idx))
    subset = Subset(traindata, ind_original)
    Xrep, labels = _compute_rep(model, subset, **param)

    clf = RFC(n_estimators=500, max_depth=6)
    clf.fit(Xrep[indtrain], labels[indtrain])
    prob = clf.predict_proba(Xrep)

    OA_train, OA_test = _compute_OA(prob, labels, indtrain, indtest)
    AA_train, AA_test = _compute_AA(prob, labels, indtrain, indtest)
    kappa_train, kappa_test = _compute_kappa(prob, labels, indtrain, indtest)
    return OA_train, AA_train, kappa_train, OA_test, AA_test, kappa_test


def evaluate_knn_clf(model, traindata, **param):
    """
    """
    ind_original = np.concatenate((traindata.train_idx, traindata.test_idx))
    indtrain = np.arange(len(traindata.train_idx))
    indtest = len(indtrain) + np.arange(len(traindata.test_idx))
    subset = Subset(traindata, ind_original)
    Xrep, labels = _compute_rep(model, subset, **param)

    clf = KNN(n_neighbors=5)
    clf.fit(Xrep[indtrain], labels[indtrain])
    prob = clf.predict_proba(Xrep)

    OA_train, OA_test = _compute_OA(prob, labels, indtrain, indtest)
    AA_train, AA_test = _compute_AA(prob, labels, indtrain, indtest)
    kappa_train, kappa_test = _compute_kappa(prob, labels, indtrain, indtest)
    return OA_train, AA_train, kappa_train, OA_test, AA_test, kappa_test


def mean_average_precision(model, traindata, **param):
    """
    """
    ind_original = np.concatenate((traindata.train_idx, traindata.test_idx))
    indtrain = np.arange(len(traindata.train_idx))
    indtest = len(indtrain) + np.arange(len(traindata.test_idx))
    subset = Subset(traindata, ind_original)
    Xrep, labels = _compute_rep(model, subset, **param)

    map_score = []
    for i in range(len(Xrep)):
        xa = Xrep[i, :]
        xb = Xrep
        dist = np.sqrt(np.sum((xb - xa)**2, axis=1)+1e-6)
        dist = np.delete(dist, i)
        prob = np.exp(-1*dist/dist.std())
        label_match = np.array((labels == labels[i])*1, dtype=np.int32)
        label_match = np.delete(label_match, i)
        map_score.append(average_precision_score(label_match, prob))
    return np.mean(map_score)
