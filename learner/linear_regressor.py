# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
from abstract_learner import AbstractRegressor
from base.dataloader import DataLoader
from base.metric import mean_error
from optimizer.sgd import StochasticGradientDescent
from loss.mean_square_error import MeanSquareError
from base.common_function import analytic_solution, normalize_data


class LinearRegressor(AbstractRegressor):
    def __init__(self, solve_type='numeric', normalize=True, max_iter=10, batch_size=10, learning_rate=0.001,
                 is_plot_loss=True):
        super(LinearRegressor, self).__init__()
        assert solve_type in ['numeric', 'analytic'], 'solve type is invalid.'
        self._nFeat = None
        self._nClass = None
        self._lossor = MeanSquareError()
        self._norm_factor = 0

        self._solve_type = solve_type
        self._normalize = normalize
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._learning_rate = learning_rate
        self._is_plot_loss = is_plot_loss

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        pred = None
        _X = X
        if self._solve_type == 'numeric':
            if self._normalize is True:
                _X = copy.deepcopy(X)
                _X = (_X - self.Xmean) / self.Xstd
            nSize = _X.shape[0]
            pred = np.dot(_X, self._parameter['weight']) + np.repeat(
                np.reshape(self._parameter['bias'], (1, self._parameter['bias'].shape[0])), _X.shape[0], axis=0)
        elif self._solve_type == 'analytic':
            pred = np.dot(_X, self._parameter['coef'])
        pred += self._norm_factor
        return pred

    def fit(self, X, y):
        _X = copy.deepcopy(X)
        _y = copy.deepcopy(y)
        assert self.__check_valid(_X, _y), 'input is invalid.'
        if self._normalize is True:
            _X, _y, self.Xmean, self.ymean, self.Xstd = normalize_data(_X, _y, inplace=True)
        if self._is_trained is False:
            self._nFeat = _X.shape[1]
        if self._solve_type == 'numeric':
            if self._is_trained is False:
                self._parameter['weight'] = np.random.uniform(0, 0.08, (self._nFeat, 1))
                self._parameter['bias'] = np.zeros(1)
                self._grad_parameter['weight'] = np.zeros((self._nFeat, 1))
                self._grad_parameter['bias'] = np.zeros(1)
            nSize = _X.shape[0]
            assert nSize >= self._batch_size, 'batch size must less or equal than X size.'
            optimizer = StochasticGradientDescent(learning_rate=self._learning_rate, batch_size=self._batch_size,
                                                  max_iter=self._max_iter, epoches_per_plot=100)
            optimizer.run(feval=self.feval, X=_X, y=_y, parameter=self._parameter)
            if self._normalize is True:
                self._norm_factor = self.ymean
            self._is_trained = True
        elif self._solve_type == 'analytic':
            self._parameter['coef'] = analytic_solution(_X, _y)
            if self._normalize is True:
                self._parameter['coef'] /= self.Xstd
                self._norm_factor = self.ymean - np.dot(self.Xmean, self._parameter['coef'])
        else:
            raise ValueError
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

    def feval(self, parameter, X, y):
        y = np.reshape(y, (y.shape[0], 1))
        nSize = X.shape[0]
        h = np.dot(X, parameter['weight']) + np.repeat(
            np.reshape(parameter['bias'], (1, parameter['bias'].shape[0])), X.shape[0], axis=0)
        loss = self._lossor.calculate(y, h)
        residual = h - y
        self._grad_parameter['weight'] = 1. / nSize * np.dot(X.T, residual)
        self._grad_parameter['bias'] = 1. / nSize * np.sum(residual)
        return loss, self._grad_parameter


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/winequality-white.csv'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='quality')
    trainset, testset = dataset.cross_split()
    linear = LinearRegressor(solve_type='numeric', normalize=True, max_iter=2000, batch_size=50,
                             learning_rate=1e-2,
                             is_plot_loss=True)
    linear.fit(trainset[0], trainset[1])
    prediction = linear.predict(testset[0])
    performance = mean_error(testset[1], prediction)
    print performance
