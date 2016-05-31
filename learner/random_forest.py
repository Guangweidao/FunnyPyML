# -*- coding: utf-8 -*-

import copy
import os
import numpy as np

from base.dataloader import DataLoader
from base.metric import accuracy_score
from learner.decision_tree import DecisionTreeClassifier
from abstract_learner import AbstractClassifier
from util.sampler import Sampler
from collections import defaultdict


class RandomForest(AbstractClassifier):
    def __init__(self, k, column_sample_rate=0.7):
        super(RandomForest, self).__init__()

        self._K = k
        self._column_sample_rate = column_sample_rate
        self._base_learner = DecisionTreeClassifier()

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nFeat = X.shape[1]
            self._class_label = list(np.unique(y))
            self._class_label.sort()
            self._nClass = len(self._class_label)
        nSize = X.shape[0]
        row_sampler = Sampler(nSize, nSize)
        col_sampler = Sampler(self._nFeat, int(np.floor(self._nFeat * self._column_sample_rate)))
        self._parameter['forest'] = [copy.deepcopy(self._base_learner) for i in xrange(self._K)]
        self._parameter['col_sample_ix'] = list()
        for i in xrange(self._K):
            row_sample_ix = row_sampler.sample(replacement=True)
            col_sample_ix = col_sampler.sample(replacement=False)
            self._parameter['col_sample_ix'].append(col_sample_ix)
            sample_X, sample_y = X[:, col_sample_ix][row_sample_ix, :], y[row_sample_ix]
            model = self._parameter['forest'][i]
            model.fit(sample_X, sample_y)
        self._is_trained = True

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nClass = len(np.unique(y))
            if nFeat == self._nFeat and nClass == self._nClass:
                is_valid = True
            return is_valid

    def predict_proba(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        nSize = X.shape[0] if len(X.shape) == 2 else 1
        pred = [np.zeros(self._nClass) for i in xrange(nSize)]
        for model, col_sample_ix in zip(self._parameter['forest'], self._parameter['col_sample_ix']):
            proba = model.predict_proba(X[:, col_sample_ix])
            for i in xrange(nSize):
                ix = np.argmax(proba[i])
                pred[i][ix] += proba[i][ix]
        return np.array(pred)

    def predict(self, X):
        pred_proba = self.predict_proba(X)
        pred = np.argmax(pred_proba, axis=1)
        pred = [self._class_label[i] for i in pred]
        return np.array(pred)


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/dataset_21_car.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='class')
    trainset, testset = dataset.cross_split()
    dt = DecisionTreeClassifier()
    dt.fit(trainset[0], trainset[1])
    predict = dt.predict(testset[0])
    print 'DecisionTree accuracy:', accuracy_score(testset[1], predict)
    rf = RandomForest(100, 0.9)
    rf.fit(trainset[0], trainset[1])
    predict = rf.predict(testset[0])
    print 'RandomForest accuracy', accuracy_score(testset[1], predict)
    # rf.dump('rf.model')
    # rf = RandomForest.load('rf.model')
    # predict = rf.predict(testset[0])
    # print 'RandomForest accuracy', accuracy_score(testset[1], predict)

