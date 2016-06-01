# -*- coding: utf-8 -*-
import numpy as np


def accuracy_score(ytrue, pred):
    # higher is better
    assert len(ytrue) == len(pred), 'inputs length must be equal.'
    nSize = len(ytrue)
    cnt = np.nansum([1 if a == b else 0 for a, b in zip(ytrue, pred)])
    return float(cnt) / nSize


def mean_error(ytrue, pred):
    # lower is better
    assert len(ytrue) == len(pred), 'inputs length must be equal.'
    return np.nanmean(np.abs(ytrue - pred))


def cluster_f_measure(ytrue, pred):
    # higher is better
    assert len(ytrue) == len(pred), 'inputs length must be equal.'
    label2ix = {label: i for i, label in enumerate(np.unique(ytrue))}
    _ytrue = np.array([label2ix[v] for v in ytrue])
    nSize = len(_ytrue)
    nClassTrue = len(np.unique(ytrue))
    nClassPred = len(np.unique(pred))
    f = np.zeros((nClassTrue, nClassPred)).astype(dtype=np.float64)
    for i in xrange(nClassTrue):
        freq_i = len(_ytrue[_ytrue == i])
        for j in xrange(nClassPred):
            freq_j = len(pred[pred == j])
            freq_i_j = float(len(filter(lambda x: x == j, pred[_ytrue == i])))
            precision = freq_i_j / freq_j if freq_j != 0 else 0
            recall = freq_i_j / freq_i if freq_i != 0 else 0
            if precision == 0 or recall == 0:
                f[i, j] = 0.
            else:
                f[i, j] = 2. * (precision * recall) / (precision + recall)
    return np.nansum([f[i][j] * len(_ytrue[_ytrue == i]) for i in xrange(nClassTrue) for j in xrange(nClassPred)]) / nSize
