# -*- coding: utf-8 -*-
import numpy as np
import copy


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
