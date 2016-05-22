# -*- coding: utf-8 -*-
import numpy as np


def accuracy_score(ytrue, pred):
    # higher is better
    assert len(ytrue) == len(pred), 'inputs length must be equal.'
    nSize = len(ytrue)
    cnt = np.sum([1 if a == b else 0 for a, b in zip(ytrue, pred)])
    return float(cnt) / nSize


def mean_error(ytrue, pred):
    # lower is better
    assert len(ytrue) == len(pred), 'inputs length must be equal.'
    return np.mean(np.abs(ytrue - pred))
