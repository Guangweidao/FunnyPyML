# -*- coding: utf-8 -*-

import math
import numpy as np
from abstract_optimizer import AbstractOptimizer


class StochasticGradientDescent(AbstractOptimizer):
    def __init__(self, learning_rate=0.01, batch_size=1000, max_iter=100, is_plot_loss=True, epoches_record_loss=10):
        super(StochasticGradientDescent, self).__init__()
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__max_iter = max_iter
        self.__is_plot_loss = is_plot_loss
        self.__epoches_record_loss = epoches_record_loss

    def optim(self, feval, X, y, parameter):
        nSize = X.shape[0]
        nBatch = int(math.ceil(float(nSize) / self.__batch_size))
        assert self.__batch_size <= nSize, 'batch size must less or equal than X size'
        ix = 0
        loss = None
        for epoch in range(self.__max_iter):
            for i in range(nBatch):
                batch = min(self.__batch_size, nSize - ix)
                _X = X[ix: ix + batch]
                _y = y[ix: ix + batch]
                ix = ix + batch if ix + batch < nSize else 0
                loss, grad_parameter = feval(parameter, _X, _y)
                parameter -= grad_parameter * self.__learning_rate
            if epoch % self.__epoches_record_loss == 0 or epoch == self.__max_iter - 1 and loss is not None:
                self.losses.append(loss)
                self._logger.info('Epoch %d\tloss: %f' % (epoch, loss))
        if self.__is_plot_loss is True:
            self.plot()
        return parameter

