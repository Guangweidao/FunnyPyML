# -*- coding: utf-8 -*-

from abc import abstractmethod


class AbstractLearner(object):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class AbstractClassifier(AbstractLearner):

    def __init__(self):
        self._parameter = dict()
        self._grad_parameter = dict()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def feval(self):
        pass


class AbstractRegressor(AbstractLearner):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self):
        pass
