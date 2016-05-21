# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from abc import abstractmethod
from base._logging import get_logger

logger = get_logger(__name__)


class AbstractOptimizer(object):
    def __init__(self):
        self.losses = list()

    @abstractmethod
    def run(self, feval, X, y, parameter):
        pass

    def plot(self):
        if len(self.losses) > 1:
            plt.plot(self.losses, hold=True)
            plt.show(block=True)

        else:
            logger.warning('loss information is too few.')
