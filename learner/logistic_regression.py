# -*- coding: utf-8 -*-

import os
import numpy as np
import math
from abstract_learner import AbstractClassifier
from base.dataloader import DataLoader


class LogisticRegression(AbstractClassifier):
    def __init__(self, dataset, max_iter=10, batch_size=10):
        assert len(dataset) == 2, 'dataset must have X adn y, so dim must be 2.'
        super(LogisticRegression, self).__init__()
        self._X = dataset[0]
        self._y = dataset[1]
        self._nSize = self._X.shape[0]
        self._nFeat = self._X.shape[1]
        self._nClass = len(np.unique(self._y))
        self._nClass = 1
        assert self._nClass == 2, 'class num must be 2.'
        assert self._nSize >= batch_size, 'batch size must less or equal than X size.'
        self._parameter['weight'] = np.random.uniform(0, 0.08, (self._nFeat, self._nClass))
        self._parameter['bias'] = np.zeros(self._nClass)
        self._grad_parameter['weight'] = np.zeros((self._nFeat, self._nClass))
        self._grad_parameter['bias'] = np.zeros(self._nClass)
        self._ix = 0
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._nBatch = math.ceil(float(self._nSize) / self._batch_size)

    def predict(self):
        super(LogisticRegression, self).predict()

    def fit(self):


    def __get_next(self):
        batch = min(self._batch_size, self._nSize - self._ix - 1)
        X = self._X[self._ix, self._ix + batch]
        y = self._y[self._ix, self._ix + batch]
        self._ix = self._ix + batch if self._ix + batch < self._nSize else 0
        return X, y

    def feval(self, parameter):
        X, y = self.__get_next()
        h = np.dot(X, parameter['weight']) + np.repeat(np.reshape(parameter['bias']), X.shape[0], axis=0)




if __name__ == '__main__':
    path = os.getcwd() + '/dataset/dataset_40_sonar.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='Class')
    trainset, testset = dataset.cross_split()
    lr = LogisticRegression(trainset)
    lr.fit(trainset[0], trainset[1])
