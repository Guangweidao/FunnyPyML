# -*- coding: utf-8 -*-
import os
import numpy as np
from abstract_learner import AbstractRegressor
from base.dataloader import DataLoader
from base.metric import mean_error


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
    path = os.getcwd() + '/../dataset/winequality-white.csv'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='quality')
    trainset, testset = dataset.cross_split()
    knn = KNNRegressor()
    knn.fit(trainset[0], trainset[1])
    predict = knn.predict(testset[0])
    print mean_error(testset[1], predict)
