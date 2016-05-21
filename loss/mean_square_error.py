# -*- coding: utf-8 -*-
import numpy as np
from abstract_loss import AbstractLoss


class MeanSquareError(AbstractLoss):
    def calculate(self, ground_truth, hypothesis):
        nSize = len(ground_truth)
        loss = 1 / (2 * nSize) * np.sum((hypothesis - ground_truth) ** 2)
        return loss
