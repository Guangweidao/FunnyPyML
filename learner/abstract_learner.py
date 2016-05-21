# -*- coding: utf-8 -*-

import cPickle
import os
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
        self._is_trained = False

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def feval(self):
        pass

    def dump(self, output_path):
        with open(output_path, 'w') as output_file:
            cPickle.dump(self, output_file)

    @staticmethod
    def load(input_path):
        assert os.path.exists(input_path), 'input path does not exist.'
        with open(input_path, 'r') as input_file:
            model = cPickle.load(input_file)
            assert isinstance(model, AbstractClassifier), 'loaded model is not what you are looking for.'
        return model




class AbstractRegressor(AbstractLearner):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self):
        pass
