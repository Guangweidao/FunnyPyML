# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from abc import abstractmethod
from base._logging import get_logger


class AbstractOptimizer(object):
    def __init__(self):
        self._logger = get_logger(self.__class__.__name__)
        self.losses = list()

    @abstractmethod
    def optim(self, feval, X, y, parameter):
        pass

    def plot(self):
        if len(self.losses) > 1:
            plt.plot(self.losses, hold=True)
            plt.show(block=True)

        else:
            self._logger.warning('loss information is too few.')
