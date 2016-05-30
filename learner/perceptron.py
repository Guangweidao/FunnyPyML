# -*- coding: utf-8 -*-
import os
import numpy as np
from abstract_learner import AbstractClassifier
from base.common_function import roll_parameter, unroll_parameter
from optimizer.sgd import StochasticGradientDescent
from base.dataloader import DataLoader
from base.metric import accuracy_score
from loss.zero_one_loss import ZeroOneLoss


class Perceptron(AbstractClassifier):
    def __init__(self, max_iter=10, learning_rate=0.001, is_plot_loss=True):
        super(Perceptron, self).__init__()
        self._nFeat = None
        self._nClass = None
        self._label2ix = None
        self._ix2label = None
        self._parameter_shape = list()
        self._lossor = ZeroOneLoss()

        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._is_plot_loss = is_plot_loss

    def fit(self, X, y):
        nSize, nFeat = X.shape
        nClass = len(np.unique(y))
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nFeat = nFeat
            self._nClass = nClass
            assert self._nClass == 2, 'class number must be 2.'
            self._label2ix = {v: i for i, v in enumerate(np.unique(y))}
            self._ix2label = {i: v for v, i in self._label2ix.iteritems()}
            W = np.random.uniform(-0.08, 0.08, (self._nFeat, 1))
            b = np.zeros(1)
            self._parameter_shape.append(W.shape)
            self._parameter_shape.append(b.shape)
            self._parameter = roll_parameter([W, b])
        y = np.array([-1 if self._label2ix[v] == 0 else 1 for v in y])
        y = np.reshape(y, (-1, 1)).astype(np.float64)
        optimizer = StochasticGradientDescent(learning_rate=self._learning_rate, batch_size=1, decay_strategy='anneal',
                                              max_iter=self._max_iter, is_plot_loss=True, add_gradient_noise=False)
        self._parameter = optimizer.optim(feval=self.feval, X=X, y=y, parameter=self._parameter)
        self._is_trained = True

    def feval(self, parameter, X, y):
        y = np.reshape(y, (-1, 1))
        param_list = unroll_parameter(parameter, self._parameter_shape)
        W, b = param_list[0], param_list[1]
        proj = np.dot(X, W) + np.repeat(np.reshape(b, (1, -1)), X.shape[0], axis=0)
        h = np.sign(proj)
        loss = self._lossor.calculate(y, h)
        grad_W = np.dot(X.T, y)
        grad_b = y
        grad_parameter = roll_parameter([grad_W, grad_b])
        return loss, grad_parameter

    def predict(self, X):
        super(Perceptron, self).predict(X)

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
    path = os.getcwd() + '/../dataset/electricity-normalized.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='Class')
    trainset, testset = dataset.cross_split()
    model = Perceptron(max_iter=200, learning_rate=0.01, is_plot_loss=True)
    model.fit(trainset[0], trainset[1])
    predict = model.predict(testset[0])
    acc = accuracy_score(testset[1], predict)
    print 'test accuracy:', acc
