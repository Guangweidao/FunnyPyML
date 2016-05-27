# -*- coding: utf-8 -*-
import math
import numpy as np
from abstract_optimizer import AbstractOptimizer


class Adadelta(AbstractOptimizer):
    def __init__(self, batch_size=1000, max_iter=100, is_plot_loss=True, epoches_record_loss=10, is_shuffle=False,
                 add_gradient_noise=False):
        super(Adadelta, self).__init__()
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
        loss = None
        rms_param = np.zeros_like(parameter)
        rms_grad = np.zeros_like(parameter)
        for epoch in range(self._max_iter):
            loss_old = loss
            if self._is_shuffle is True:
                indices = np.random.permutation(nSize)  # shuffle the input every epoch
            else:
                indices = range(nSize)

            for i in range(nBatch):
                batch = min(self._batch_size, nSize - ix)
                _X = X[indices[ix: ix + batch]]
                _y = y[indices[ix: ix + batch]]
                ix = ix + batch if ix + batch < nSize else 0
                loss, grad_parameter = feval(parameter, _X, _y)
                if self._add_gradient_noise is True:
                    grad_parameter = self._gradient_noise(grad_parameter, 1e-3, epoch)
                rms_grad = self._gama * rms_grad + (1 - self._gama) * grad_parameter ** 2
                parameter -= np.sqrt(rms_param + self._epsilon) / np.sqrt(rms_grad + self._epsilon) * grad_parameter
                rms_param = self._gama * rms_param + (1 - self._gama) * grad_parameter ** 2
            if epoch % self._epoches_record_loss == 0 or epoch == self._max_iter - 1 and loss is not None:
                self.losses.append(loss)
                self._logger.info('Epoch %d\tloss: %f' % (epoch, loss))
            if self._check_converge(g=grad_parameter, d=-grad_parameter, loss=loss, loss_old=loss_old):
                break
        if self._is_plot_loss is True:
            self.plot()
        return parameter
