# -*- coding: utf-8 -*-
import math


class StochasticGradientDescent(object):
    def __init__(self, feval, parameter, learning_rate=0.01):
        self.__feval = feval
        self.__parameter = parameter
        self.learning_rate = learning_rate

    def run(self):
        grad_parameter = self.__feval(self.__parameter)
        self.__parameter['weight'] -= grad_parameter['weight'] * self.learning_rate
        self.__parameter['bias'] -= grad_parameter['bias'] * self.learning_rate
