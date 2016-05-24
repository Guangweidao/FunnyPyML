# -*- coding: utf-8 -*-
import numpy as np
import copy
from util.freq_dict import FreqDict


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_inv(x):
    sigm_v = sigmoid(x)
    return sigm_v * (1 - sigm_v)


def analytic_solution(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


def normalize_data(X, y, inplace=False):
    if inplace is True:
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)
    Xmean = np.mean(X, axis=0)
    X -= Xmean
    Xstd = np.sqrt(np.sum(X ** 2, axis=0))
    Xstd[Xstd == 0] = 1
    X /= Xstd
    ymean = np.mean(y)
    y = y - ymean
    return X, y, Xmean, ymean, Xstd


def euclidean_distance(x, y):
    x, y = np.array(x), np.array(y)
    assert x.shape == y.shape and len(x.shape) == 1, 'input is invalid.'
    return np.linalg.norm(x - y)


def entropy(x):
    nSize = len(x)
    fd = FreqDict(list(x))
    result = 0.
    for v in fd.values():
        prob = float(v) / nSize
        result += -prob * np.log(prob)
    return result


def condition_entropy(x, cond):
    assert x.shape == cond.shape, 'input is invalid.'
    nSize = len(x)
    fd = FreqDict(list(cond))
    fd = {k: float(v) / nSize for k, v in fd}
    result = 0.
    for k, v in fd.iteritems():
        result += v * entropy(x[cond == k])
    return result
