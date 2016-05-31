# -*- coding: utf-8 -*-
import math
import numpy as np
from abstract_optimizer import AbstractOptimizer


class Adam(AbstractOptimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.01, batch_size=1000, max_iter=100,
                 is_plot_loss=True, epoches_record_loss=10, is_shuffle=False, add_gradient_noise=False):
        super(Adam, self).__init__()
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._is_plot_loss = is_plot_loss
        self._epoches_record_loss = epoches_record_loss
        self._is_shuffle = is_shuffle
        self._add_gradient_noise = add_gradient_noise
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def optim(self, feval, X, y, parameter):
        nSize = X.shape[0]
        nBatch = int(math.ceil(float(nSize) / self._batch_size))
        assert self._batch_size <= nSize, 'batch size must less or equal than X size'
        ix = 0
        loss_new = None
        learning_rate = self._learning_rate
        m = np.zeros_like(parameter)
        v = np.zeros_like(parameter)
        beta1_t = 1
        beta2_t = 1
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
                m = self._beta1 * m + (1 - self._beta1) * grad_parameter
                v = self._beta2 * v + (1 - self._beta2) * grad_parameter ** 2
                beta1_t, beta2_t = beta1_t * self._beta1, beta2_t * self._beta2
                bias_correction_m = m / (1 - beta1_t)
                bias_correction_v = v / (1 - beta2_t)
                parameter -= 1. / (np.sqrt(bias_correction_v) + self._epsilon) * learning_rate * bias_correction_m
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
