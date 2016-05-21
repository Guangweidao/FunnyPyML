# -*- coding: utf-8 -*-
from abc import abstractmethod


class AbstractLoss(object):
    @abstractmethod
    def calculate(self, ground_truth, hypothesis):
        pass
