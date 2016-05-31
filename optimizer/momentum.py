# -*- coding: utf-8 -*-
import numpy as np
import math
from abstract_optimizer import AbstractOptimizer


class MomentumSGD(AbstractOptimizer):
    def __init__(self, learning_rate=0.01, batch_size=1000, momentum=0.9, momentum_type='nesterov', max_iter=100,
                 is_plot_loss=True, epoches_record_loss=10, is_shuffle=False, add_gradient_noise=False):
        super(MomentumSGD, self).__init__()
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._momentum = momentum
        self._momentum_type = momentum_type
        self._max_iter = max_iter
        self._is_plot_loss = is_plot_loss
        self._epoches_record_loss = epoches_record_loss
        self._is_shuffle = is_shuffle
        self._add_gradient_noise = add_gradient_noise

    def optim(self, feval, X, y, parameter):
        nSize = X.shape[0]
        nBatch = int(math.ceil(float(nSize) / self._batch_size))
        assert self._batch_size <= nSize, 'batch size must less or equal than X size'
        ix = 0
        loss_new = None
        v = 0.
        learning_rate = self._learning_rate
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
                if self._momentum_type == 'nesterov':
                    loss, grad_parameter = feval(parameter - self._momentum * v, _X, _y)
                    if self._add_gradient_noise is True:
                        grad_parameter = self._gradient_noise(grad_parameter, learning_rate, epoch)
                    v = self._momentum * v + grad_parameter * learning_rate
                    parameter -= v
                elif self._momentum_type == 'standard':
                    loss, grad_parameter = feval(parameter, _X, _y)
                    if self._add_gradient_noise is True:
                        grad_parameter = self._gradient_noise(grad_parameter, learning_rate, epoch)
                    v = self._momentum * v + grad_parameter * learning_rate
                    parameter -= v
                else:
                    raise ValueError
                loss_batches.append(loss)
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
