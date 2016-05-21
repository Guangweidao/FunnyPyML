# -*- coding: utf-8 -*-

import math
from abstract_optimizer import AbstractOptimizer
from base._logging import get_logger

logger = get_logger(__name__)


class StochasticGradientDescent(AbstractOptimizer):
    def __init__(self, learning_rate=0.01, batch_size=1000, max_iter=100, epoches_per_plot=10, epoches_record_loss=10):
        super(StochasticGradientDescent, self).__init__()
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__max_iter = max_iter
        self.__epoches_per_plot = epoches_per_plot
        self.__epoches_record_loss = epoches_record_loss

    def run(self, feval, X, y, parameter):
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
                parameter['weight'] -= grad_parameter['weight'] * self.__learning_rate
                parameter['bias'] -= grad_parameter['bias'] * self.__learning_rate
            if epoch % self.__epoches_record_loss == 0 or epoch == self.__max_iter - 1 and loss is not None:
                self.losses.append(loss)
                logger.info('Epoch %d\tloss: %f' % (epoch, loss))
                # I does not find a way to plot loss every epoches, because plt.show will block program
                # if epoch % self.__epoches_per_plot == 0 and epoch != 0:
                #     self.plot()
        self.plot()
