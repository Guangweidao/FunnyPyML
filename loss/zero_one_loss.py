# -*- coding: utf-8 -*-
import numpy as np
from abstract_loss import AbstractLoss


class ZeroOneLoss(AbstractLoss):
    def calculate(self, ground_truth, hypothesis):
        ground_truth = ground_truth.as_type(np.int)
        hypothesis = hypothesis.as_type(np.int)
        loss = np.sum(ground_truth ^ hypothesis)
        return loss
