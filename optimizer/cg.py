# -*- coding: utf-8 -*-
import numpy as np
import functools
import copy
from abstract_optimizer import AbstractOptimizer
from util.line_search import wolfe


class ConjugateGradientDescent(AbstractOptimizer):
    def __init__(self, max_iter=100, is_plot_loss=True, epoches_record_loss=10):
        super(ConjugateGradientDescent, self).__init__()
        self.__max_iter = max_iter
        self.__is_plot_loss = is_plot_loss
        self.__epoches_record_loss = epoches_record_loss
        self._tol = 1e-9
        self._eps = 1e-6

    def optim(self, feval, X, y, parameter):
        nSize = X.shape[0]
        f = functools.partial(feval, X=X, y=y)
        loss = None
        epoch = 0
        k = 0  # restart factor, restart whenever a search direction is not a descent direction
        loss, g = f(parameter)
        r = -g
        d = r
        delta_new = np.dot(r.T, r)
        delta_zero = delta_new
        while epoch < self.__max_iter and delta_new > self._eps * self._eps * delta_zero:
            gtd_new = np.dot(g.T, d)
            if gtd_new > - self._tol:
                self._logger.info('directional derivative below tol.')
                break
            if epoch == 0:
                alpha = min(1, 1 / np.sum(np.abs(g)))
            else:
                alpha *= min(2, gtd_old / gtd_new)
            gtd_old = gtd_new
            loss_old = loss
            alpha, loss, g, parameter = wolfe(f, parameter, d, g, loss_old, alpha)
            if np.abs(loss - loss_old) < self._tol:
                self._logger.info('loss changing by less than tol.')
                break
            r = -g
            delta_old = delta_new
            delta_new = np.dot(r.T, r)
            beta = delta_new / delta_old
            d = r + beta * d
            if np.sum(np.abs(alpha * d)) < self._tol:
                self._logger.info('step size below tol.')
                break
            k += 1
            if k == 50 or np.dot(r.T, d) <= 0:
                d = r
                k = 0
            if epoch % self.__epoches_record_loss == 0 or epoch == self.__max_iter - 1 and loss is not None:
                self.losses.append(loss)
                self._logger.info('Epoch %d\tloss: %f' % (epoch, loss))
            epoch += 1
        if self.__is_plot_loss is True:
            self.plot()
        return parameter
