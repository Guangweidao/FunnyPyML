# -*- coding: utf-8 -*-

import os
import numpy as np
import math
from abstract_learner import AbstractClassifier
from base.dataloader import DataLoader
from base.common_function import *
from optimizer.sgd import StochasticGradientDescent
from base.metric import accuracy_score


class LogisticRegression(AbstractClassifier):
    def __init__(self, dataset, max_iter=10, batch_size=10):
        assert len(dataset) == 2, 'dataset must have X adn y, so dim must be 2.'
        super(LogisticRegression, self).__init__()
        self._X = dataset[0]
        self._label2ix = {label: i for i, label in enumerate(np.unique(dataset[1]))}
        self._ix2label = {i: label for i, label in enumerate(np.unique(dataset[1]))}
        self._y = np.array([self._label2ix[label] for label in dataset[1]])
        self._nSize = self._X.shape[0]
        self._nFeat = self._X.shape[1]
        self._nClass = len(np.unique(self._y))
        assert self._nClass == 2, 'class num must be 2.'
        self._nClass = 1
        assert self._nSize >= batch_size, 'batch size must less or equal than X size.'
        self._parameter['weight'] = np.random.uniform(0, 0.08, (self._nFeat, self._nClass))
        self._parameter['bias'] = np.zeros(self._nClass)
        self._grad_parameter['weight'] = np.zeros((self._nFeat, self._nClass))
        self._grad_parameter['bias'] = np.zeros(self._nClass)
        self._ix = 0
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._nBatch = math.ceil(float(self._nSize) / self._batch_size)

    def predict(self, X):
        nSize = X.shape[0]
        proj = np.dot(X, self._parameter['weight']) + np.repeat(
            np.reshape(self._parameter['bias'], (1, self._parameter['bias'].shape[0])), X.shape[0], axis=0)
        h = sigmoid(proj)
        pred = [1 if v >= 0.5 else 0 for v in h]
        return np.array([self._ix2label[ix] for ix in pred])

    def fit(self):
        optimizer = StochasticGradientDescent(self.feval, self._parameter, learning_rate=0.01)
        for epoch in range(self._max_iter):
            optimizer.run()

    def __get_next(self):
        batch = min(self._batch_size, self._nSize - self._ix)
        assert batch > 0, 'batch input must be larger than 0.'
        X = self._X[self._ix: self._ix + batch]
        y = self._y[self._ix: self._ix + batch]
        self._ix = self._ix + batch if self._ix + batch < self._nSize else 0
        return X, y

    def feval(self, parameter):
        X, y = self.__get_next()
        y = np.reshape(y, (y.shape[0], 1))
        nSize = X.shape[0]
        proj = np.dot(X, parameter['weight']) + np.repeat(
            np.reshape(parameter['bias'], (1, parameter['bias'].shape[0])), X.shape[0], axis=0)
        h = sigmoid(proj)
        self._grad_parameter['weight'] = 1. / nSize * np.dot(X.T, (h - y))
        self._grad_parameter['bias'] = 1. / nSize * np.sum(h - y)
        return self._grad_parameter


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/dataset_40_sonar.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='Class')
    trainset, testset = dataset.cross_split()
    lr = LogisticRegression(trainset, max_iter=1000)
    lr.fit()
    predict = lr.predict(testset[0])
    acc = accuracy_score(testset[1], predict)
    print predict
    print 'accuracy:', acc
