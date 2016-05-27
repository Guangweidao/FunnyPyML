# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from base._logging import get_logger


class AbstractOptimizer(object):
    def __init__(self):
        self._logger = get_logger(self.__class__.__name__)
        self.losses = list()
        self._tol = 1e-9

    @abstractmethod
    def optim(self, feval, X, y, parameter):
        pass

    def _check_converge(self, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else None
        g = kwargs['g'] if 'g' in kwargs else None
        d = kwargs['d'] if 'd' in kwargs else None
        gtd = np.dot(g.T, d) if g is not None else None
        loss = kwargs['loss'] if 'loss' in kwargs else None
        loss_old = kwargs['loss_old'] if 'loss_old' in kwargs else None
        is_converge = False
        if gtd is not None and gtd > - self._tol:
            self._logger.info('directional derivative below tol.')
            is_converge = True
        elif g is not None and alpha is not None and np.sum(np.abs(alpha * d)) < self._tol:
            self._logger.info('step size below tol.')
            is_converge = True
        elif loss is not None and loss_old is not None and np.abs(loss - loss_old) < self._tol:
            self._logger.info('loss changing by less than tol.')
            is_converge = True
        return is_converge

    def _gradient_noise(self, g, alpha, epoch):
        gamma = 3
        noise = np.random.normal(0, np.sqrt(alpha / (1 + epoch) ** gamma), g.shape)
        return g + noise

    def plot(self):
        if len(self.losses) > 1:
            plt.plot(self.losses, hold=True)
            plt.show(block=True)

        else:
            self._logger.warning('loss information is too few.')
