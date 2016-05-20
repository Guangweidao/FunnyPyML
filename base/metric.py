# -*- coding: utf-8 -*-
import numpy as np


def accuracy_score(ytrue, pred):
    assert len(ytrue) == len(pred), 'inputs length must be equal.'
    nSize = len(ytrue)
    cnt = np.sum([1 if a == b else 0 for a, b in zip(ytrue, pred)])
    return float(cnt) / nSize
