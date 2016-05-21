# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_inv(x):
    sigm_v = sigmoid(x)
    return sigm_v * (1 - sigm_v)
