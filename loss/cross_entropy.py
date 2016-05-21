# -*- coding: utf-8 -*-
from abstract_loss import AbstractLoss
import numpy as np


class CrossEntropy(AbstractLoss):
    def calculate(self, ground_truth, hypothesis):
        nSize = len(ground_truth)
        loss = -1. / nSize * np.sum(ground_truth * np.log(hypothesis) + (1 - ground_truth) * np.log(1 - hypothesis))
        return loss
