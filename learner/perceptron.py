# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from abstract_learner import AbstractClassifier
from base.common_function import roll_parameter, unroll_parameter
from optimizer.sgd import StochasticGradientDescent
from base.dataloader import DataLoader
from base.metric import accuracy_score
from loss.zero_one_loss import ZeroOneLoss
from base._logging import get_logger


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
        optimizer = StochasticGradientDescent(learning_rate=self._learning_rate, batch_size=1, max_iter=self._max_iter,
                                              is_plot_loss=True, add_gradient_noise=False)
        self._parameter = optimizer.optim(feval=self.feval, X=X, y=y, parameter=self._parameter)
        self._is_trained = True

    def feval(self, parameter, X, y):
        y = np.reshape(y, (-1, 1))
        param_list = unroll_parameter(parameter, self._parameter_shape)
        W, b = param_list[0], param_list[1]
        proj = np.dot(X, W) + np.repeat(np.reshape(b, (1, -1)), X.shape[0], axis=0)
        h = np.sign(proj)
        loss = self._lossor.calculate(y, h)
        if loss > 0:
            grad_W = -np.dot(X.T, y)
            grad_b = -np.sum(y)
        else:
            grad_W = np.zeros(W.shape)
            grad_b = np.zeros(b.shape)
        grad_parameter = roll_parameter([grad_W, grad_b])
        return loss, grad_parameter

    def predict(self, X):
        pred = self._predict(X)
        pred = [1 if p == 1 else 0 for p in pred]
        return np.array([self._ix2label[v] for v in pred])

    def _predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        param_list = unroll_parameter(self._parameter, self._parameter_shape)
        W, b = param_list[0], param_list[1]
        proj = np.dot(X, W) + np.repeat(np.reshape(b, (1, -1)), X.shape[0], axis=0)
        pred = np.sign(proj)
        return pred

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

    def plot(self, X):
        nSize, nFeat = X.shape
        if nFeat != 2:
            logger.warning('feature number must be 2.')
            return
        logger.info('start plotting...')
        pred = self._predict(X)
        h = 0.02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self._predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=plt.cm.Paired)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        plt.show()

logger = get_logger(Perceptron.__name__)

if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/iris.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='Class')
    trainset, testset = dataset.cross_split()
    model = Perceptron(max_iter=200, learning_rate=0.01, is_plot_loss=True)
    model.fit(trainset[0][:, [0, 2]], trainset[1])
    predict = model.predict(testset[0][:, [0, 2]])
    acc = accuracy_score(testset[1], predict)
    print 'test accuracy:', acc
    model.plot(trainset[0][:, [0, 2]])

#