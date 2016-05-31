# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
from abstract_learner import AbstractRegressor
from base.dataloader import DataLoader
from decision_tree import DecisionTreeRegressor
from base.metric import mean_error


class GradientBoostingDecisionTree(AbstractRegressor):
    def __init__(self, k=10, tol=0):
        super(GradientBoostingDecisionTree, self).__init__()
        self._K = k
        self._tol = tol
        self._base_learner = DecisionTreeRegressor()

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        nSize, nFeat = X.shape
        if self._is_trained is False:
            self._nFeat = nFeat
        self._parameter['trees'] = list()
        fxi = np.zeros(y.shape)
        residual = y
        for i in xrange(self._K):
            residual = residual - fxi
            if np.abs(np.mean(residual)) < self._tol:
                break
            model = copy.deepcopy(self._base_learner)
            model.fit(X, residual)
            fxi = model.predict(X)
            self._parameter['trees'].append(model)
        self._is_trained = True

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            if nFeat == self._nFeat:
                is_valid = True
            return is_valid

    def predict(self, X):
        models = self._parameter['trees']
        pred = np.zeros(X.shape[0])
        for model in models:
            pred += np.array(model.predict(X))
        return pred


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/winequality-white.csv'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='quality')
    trainset, testset = dataset.cross_split()
    gbdt = GradientBoostingDecisionTree(10)
    gbdt.fit(trainset[0], trainset[1])
    predict = gbdt.predict(testset[0])
    print 'GBDT mean error:', mean_error(testset[1], predict)

    dt = DecisionTreeRegressor()
    dt.fit(trainset[0], trainset[1])
    predict = dt.predict(testset[0])
    print 'DecisionTree mean error:', mean_error(testset[1], predict)

