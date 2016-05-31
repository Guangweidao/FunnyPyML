# -*- coding: utf-8 -*-

import math
import numpy as np
from abstract_optimizer import AbstractOptimizer


class RMSProp(AbstractOptimizer):
    def __init__(self, learning_rate=0.01, batch_size=1000, max_iter=100, is_plot_loss=True, epoches_record_loss=10,
                 is_shuffle=False, add_gradient_noise=False):
        super(RMSProp, self).__init__()
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._is_plot_loss = is_plot_loss
        self._epoches_record_loss = epoches_record_loss
        self._is_shuffle = is_shuffle
        self._add_gradient_noise = add_gradient_noise
        self._gama = 0.9
        self._epsilon = 1e-8

    def optim(self, feval, X, y, parameter):
        nSize = X.shape[0]
        nBatch = int(math.ceil(float(nSize) / self._batch_size))
        assert self._batch_size <= nSize, 'batch size must less or equal than X size'
        ix = 0
        loss_new = None
        learning_rate = self._learning_rate
        rms_grad = np.zeros_like(parameter)
        for epoch in range(self._max_iter):
            loss_old = loss_new
            if self._is_shuffle is True:
                indices = np.random.permutation(nSize)  # shuffle the input every epoch
            else:
                indices = range(nSize)
            loss_batches = list()
            for i in range(nBatch):
                batch = min(self._batch_size, nSize - ix)
                _X = X[indices[ix: ix + batch]]
                _y = y[indices[ix: ix + batch]]
                ix = ix + batch if ix + batch < nSize else 0
                loss, grad_parameter = feval(parameter, _X, _y)
                loss_batches.append(loss)
                if self._add_gradient_noise is True:
                    grad_parameter = self._gradient_noise(grad_parameter, learning_rate, epoch)
                rms_grad = self._gama * rms_grad + (1 - self._gama) * grad_parameter ** 2
                parameter -= 1. / np.sqrt(rms_grad + self._epsilon) * grad_parameter * learning_rate
            loss_new = np.mean(loss_batches)
            if epoch % self._epoches_record_loss == 0 or epoch == self._max_iter - 1 and loss_new is not None:
                self.losses.append(loss_new)
                self._logger.info('Epoch %d\tloss: %f' % (epoch, loss_new))
            if self._check_converge(g=grad_parameter, d=-grad_parameter, loss=loss_new, loss_old=loss_old,
                                    alpha=learning_rate):
                break
        if self._is_plot_loss is True:
            self.plot()
        return parameter
