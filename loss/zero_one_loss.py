# -*- coding: utf-8 -*-
import numpy as np
from abstract_loss import AbstractLoss


class ZeroOneLoss(AbstractLoss):
    def calculate(self, ground_truth, hypothesis):
        loss = np.sum([1 if a != b else 0 for a, b in zip(ground_truth, hypothesis)])
        return loss
