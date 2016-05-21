# -*- coding: utf-8 -*-
import numpy as np
from abstract_learner import AbstractClassifier


class KNNClassifier(AbstractClassifier):
    def __init__(self):
        super(KNNClassifier, self).__init__()

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nDim = len(X.shape)
            self._nFeat = X.shape[1]
            self._parameter['neighbor'] = X
        else:
            self._parameter['neighbor'] = np.concatenate([self._parameter['neighbor'], X], axis=0)
        self._is_trained = True

    def predict(self, X):
        pred = list()
        for i in range(X.shape[0]):
            dist = list()
            for irow in range(self._parameter['neighbor'].shape[0]):
                dist.append(np.linalg.norm(X[i, :] - self._parameter['neighbor'][irow, :]))
            pass

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
