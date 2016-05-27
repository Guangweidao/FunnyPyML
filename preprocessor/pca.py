# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from base.dataloader import DataLoader


class PCA(object):
    def __init__(self, K=None, ratio=0, whiten=False):
        self._K = K
        self._ratio = ratio
        self._whiten = whiten
        self._epsilon = 1e-6
        self._mean = None
        self._V = None
        self._S = list()

    def fit(self, X):
        nSize = X.shape[0]
        self._mean = np.mean(X, axis=0)
        _X = X - self._mean
        C = 1. / nSize * np.dot(_X.T, _X)
        U, S, V = np.linalg.svd(C, full_matrices=False)
        if self._K is not None:
            K = self._K
        else:
            K = X.shape[1]
        if self._ratio > 0:
            total = np.sum(S)
            part = 0.
            for i, v in enumerate(S):
                part += v
                if part / total >= self._ratio:
                    break
            K = i + 1 if self._K is None else max([self._K, i + 1])
        self._V = V[:K, :]
        self._S = np.array(S[:K])

    def transform(self, X):
        assert self._V is not None, 'model has not yet be fitted.'
        _X = X - self._mean
        _X_transform = np.dot(_X, self._V.T)
        if self._whiten is True:
            _X_transform /= np.sqrt(self._S + self._epsilon)
        return _X_transform

    def information_distribution(self, cumulative=False, percent=False):
        if percent is True:
            info = self._S / np.sum(self._S)
        else:
            info = self._S
        if cumulative is False:
            return info
        else:
            return np.cumsum(info)

    def plot(self, X):
        nSize = X.shape[0]
        nFeat = X.shape[1]
        assert nFeat == 2, 'feature number should be 2.'
        for i in xrange(nSize):
            plt.plot(X[i, 0], X[i, 1], 'or')
        plt.show()


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/iris.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='binaryClass')
    trainset, testset = dataset.cross_split()
    pca = PCA(2, whiten=False)
    X = trainset[0][:, [0, 2]]
    pca.fit(X)
    _X = pca.transform(X)
    print pca.information_distribution(percent=True)
    pca.plot(X)
    pca.plot(_X)
