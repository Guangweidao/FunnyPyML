# -*- coding: utf-8 -*-
import os
import numpy as np
from abstract_learner import AbstractClassifier
from util.freq_dict import FreqDict
from base.dataloader import DataLoader
from base.metric import accuracy_score
from abstract_learner import AbstractRegressor
from base.metric import mean_error
from util.kd_tree import KDTree, KDNode
from base.common_function import euclidean_distance


class KNNClassifier(AbstractClassifier):
    def __init__(self, k=10, search_mode='kd_tree'):
        super(KNNClassifier, self).__init__()
        self._K = k
        self._search_mode = search_mode

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nDim = len(X.shape)
            self._nFeat = X.shape[1]
        if self._search_mode == 'kd_tree':
            self._parameter['kd_tree'] = KDTree(X, y, euclidean_distance)
        elif self._search_mode == 'brutal':
            self._parameter['neighbor_X'] = X
            self._parameter['neighbor_y'] = y
        else:
            raise ValueError
        self._is_trained = True

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        pred = list()
        if self._search_mode == 'kd_tree':
            kd_tree = self._parameter['kd_tree']
            for i in xrange(X.shape[0]):
                nn = kd_tree.search(kd_tree.root, X[i, :], self._K)
                pass

        for i in range(X.shape[0]):
            dist = list()
            for irow in range(self._parameter['neighbor_X'].shape[0]):
                dist.append(np.linalg.norm(X[i, :] - self._parameter['neighbor_X'][irow, :]))
            indices = np.argsort(dist)[:min(self._K, len(self._parameter['neighbor_y']))]
            fd = FreqDict(list(self._parameter['neighbor_y'][indices]), reverse=True)
            pred.append(fd.keys()[0])
        return pred

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nDim = len(X.shape)
            if nFeat == self._nFeat and nDim == self._nDim:
                is_valid = True
            return is_valid


class KNNRegressor(AbstractRegressor):
    def __init__(self, k=10):
        super(KNNRegressor, self).__init__()
        self._K = k

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nDim = len(X.shape)
            self._nFeat = X.shape[1]
            self._parameter['neighbor_X'] = X
            self._parameter['neighbor_y'] = y
        else:
            self._parameter['neighbor_X'] = np.concatenate([self._parameter['neighbor_X'], X], axis=0)
            self._parameter['neighbor_y'] = np.concatenate(self._parameter['neighbor_y'], y)
        self._is_trained = True

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nDim = len(X.shape)
            if nFeat == self._nFeat and nDim == self._nDim:
                is_valid = True
            return is_valid

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        pred = list()
        for i in range(X.shape[0]):
            dist = list()
            for irow in range(self._parameter['neighbor_X'].shape[0]):
                dist.append(np.linalg.norm(X[i, :] - self._parameter['neighbor_X'][irow, :]))
            indices = np.argsort(dist)[:min(self._K, len(self._parameter['neighbor_y']))]
            pred.append(np.mean(self._parameter['neighbor_y'][indices]))
        return pred


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/electricity-normalized.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='Class')
    trainset, testset = dataset.cross_split()
    knn = KNNClassifier(k=10)
    knn.fit(trainset[0], trainset[1])
    predict = knn.predict(testset[0])
    acc = accuracy_score(testset[1], predict)
    print acc
    # path = os.getcwd() + '/../dataset/winequality-white.csv'
    # loader = DataLoader(path)
    # dataset = loader.load(target_col_name='quality')
    # trainset, testset = dataset.cross_split()
    # knn = KNNRegressor()
    # knn.fit(trainset[0], trainset[1])
    # predict = knn.predict(testset[0])
    # print mean_error(testset[1], predict)
