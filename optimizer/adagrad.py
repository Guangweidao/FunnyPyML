# -*- coding: utf-8 -*-

from abstract_optimizer import AbstractOptimizer

class Adagrad(AbstractOptimizer):

    def __init__(self):
        super(Adagrad, self).__init__()


    def optim(self, feval, X, y, parameter):
        pass