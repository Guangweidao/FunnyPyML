# -*- coding: utf-8 -*-
import copy
import os
import numpy as np
from abstract_learner import AbstractClassifier
from collections import defaultdict
from base.dataloader import DataLoader
from base.metric import accuracy_score
from util.sampler import Sampler
from learner.decision_tree import DecisionTreeClassifier
from learner.logistic_regression import LogisticRegression
from learner.naive_bayes import NaiveBayes


class AdaBoost(AbstractClassifier):
    def __init__(self, base_learner, k):
        super(AdaBoost, self).__init__()
        self._base_learner = base_learner
        self._K = k
        self._parameter['model'] = [copy.deepcopy(self._base_learner) for i in xrange(self._K)]
        self._parameter['alpha'] = list()

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        nSize = X.shape[0]
        sampler = Sampler(nSize, nSize, mode='fast')
        if self._is_trained is False:
            self._nFeat = X.shape[1]
            self._nClass = len(np.unique(y))
            samples_weight = np.ones(nSize) * (1. / nSize)
        for i in xrange(self._K):
            model = self._parameter['model'][i]
            if i == 1:
                sample_ix = np.array([i for i in xrange(nSize)])
            else:
                sample_ix = sampler.sample(samples_weight)
            sample_X, sample_y = X[sample_ix], y[sample_ix]
            model.fit(sample_X, sample_y)
            pred = model.predict(X)
            diff = np.array([1 if t != p else 0 for t, p in zip(y, pred)])
            error = np.sum(diff * samples_weight)
            alpha = 1. / 2 * np.log((1 - error) / error)
            self._parameter['alpha'].append(alpha)
            diff = np.array([-1 if t != p else 1 for t, p in zip(y, pred)])
            samples_weight *= np.exp(-error * diff)
            samples_weight /= np.sum(samples_weight)
        self._is_trained = True

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        nSize = X.shape[0] if len(X.shape) == 2 else 1
        pred = [defaultdict(float) for i in xrange(nSize)]
        for alpha, model in zip(self._parameter['alpha'], self._parameter['model']):
            p = model.predict(X)
            for i, v in enumerate(p):
                pred[i][v] += alpha
        for i in xrange(nSize):
            max_prob = None
            max_label = None
            for k, v in pred[i].iteritems():
                if v > max_prob or max_prob is None:
                    max_prob = v
                    max_label = k
            pred[i] = max_label
        return np.array(pred)

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


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/dataset_21_car.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='class')
    trainset, testset = dataset.cross_split()
    nb = NaiveBayes()
    nb.fit(trainset[0], trainset[1])
    p1 = nb.predict(testset[0])
    print 'NaiveBayes accuracy:', accuracy_score(testset[1], p1)
    base_learner = NaiveBayes()
    ada = AdaBoost(base_learner, 100)
    ada.fit(trainset[0], trainset[1])
    prediction = ada.predict(testset[0])
    performance = accuracy_score(testset[1], prediction)
    print 'AdaBoost accuracy:', performance
    # ada.dump('ada.model')
    # ada = AdaBoost.load('ada.model')
    # prediction = ada.predict(testset[0])
    # performance = accuracy_score(testset[1], prediction)
    # print performance
