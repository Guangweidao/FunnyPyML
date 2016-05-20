# -*- coding: utf-8 -*-
import numpy as np


class MeanSquareError(object):
    def calculate(self, ytrue, pred):
        loss = np.sum((pred - ytrue) ** 2)
        return loss
