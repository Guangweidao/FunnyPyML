# -*- coding: utf-8 -*-
import functools
import numpy as np
from abstract_optimizer import AbstractOptimizer
from util.line_search import *


class LBFGS(AbstractOptimizer):
    def __init__(self, correction=10, max_iter=100, is_plot_loss=True, epoches_record_loss=10):
        super(LBFGS, self).__init__()
        self._correction = correction
        self._max_iter = max_iter
        self._is_plot_loss = is_plot_loss
        self._epoches_record_loss = epoches_record_loss
        self._tol = 1e-9

    def optim(self, feval, X, y, parameter):
        f = functools.partial(feval, X=X, y=y)
        loss, gk = f(parameter)
        dk = -gk
        directions = np.zeros((len(gk), 0))
        steps = np.zeros((len(gk), 0))
        gktd = np.dot(gk.T, dk)
        alpha = min(1, 1 / np.sum(np.abs(gk)))
        alpha, loss, gk_plus, parameter = wolfe(f, parameter, dk, alpha, sigma=0.9, max_iter=10)
        for epoch in xrange(self._max_iter):
            loss_old = loss
            dk, directions, steps = self._lbfgs_update(gk, gk_plus, dk, alpha, directions, steps)
            gk, gktd_plus = gk_plus, np.dot(gk_plus.T, dk)
            if gktd_plus > - self._tol:
                self._logger.info('directional derivative below tol.')
                break
            alpha = alpha * min(2, gktd / gktd_plus)
            gktd = gktd_plus
            alpha, loss, gk_plus, parameter = wolfe(f, parameter, dk, alpha)
            if np.sum(np.abs(alpha * dk)) <= self._tol:
                self._logger.info('step size below tol.')
                break
            if np.abs(loss - loss_old) < self._tol:
                self._logger.info('loss changing by less than tol.')
                break
            if epoch % self._epoches_record_loss == 0 or epoch == self._max_iter - 1 and loss is not None:
                self.losses.append(loss)
                self._logger.info('Epoch %d\tloss: %f' % (epoch, loss))
        if self._is_plot_loss is True:
            self.plot()
        return parameter

    def _lbfgs_update(self, gk, gk_plus, dk, alpha, directions, steps):
        y = gk_plus - gk
        s = alpha * dk
        directions, steps, Hk = self._limit_memory(y, s, directions, steps)
        dk_plus = self._bfgs_update(Hk, directions, steps, -gk_plus)
        return dk_plus, directions, steps

    def _limit_memory(self, y, s, direction, step):
        yts = np.dot(y.T, s)
        nCorr = direction.shape[1]
        if nCorr < self._correction:
            direction = np.hstack([direction, s.reshape(-1, 1)])
            step = np.hstack([step, y.reshape(-1, 1)])
        else:
            direction = np.hstack([direction[:, 1:], s.reshape(-1, 1)])
            step = np.hstack([step[:, 1:], y.reshape(-1, 1)])
        Hk = yts / np.dot(y.T, y)
        return direction, step, Hk

    def _bfgs_update(self, Hk, s, y, d):
        r = d
        k = s.shape[1]
        rho = np.zeros(k)
        for i in xrange(k):
            rho[i] = 1. / np.dot(y[:, i].T, s[:, i])
        alpha = np.zeros((k, 1))
        for i in xrange(k - 1, -1, -1):
            alpha[i] = rho[i] * np.dot(s[:, i].T, r)
            r = r - alpha[i] * y[:, i]
        r *= Hk
        for i in xrange(k):
            beta = rho[i] * np.dot(y[:, i].T, r)
            r = r + (alpha[i] - beta) * s[:, i]
        return r
