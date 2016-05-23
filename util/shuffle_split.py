# -*- coding: utf-8 -*-
import math
import numpy as np


class ShuffleSpliter(object):
    def __init__(self, n, ratio=0.1):
        assert ratio < 1, 'ratio is invalid.'
        self._N = n
        self._iter = iter
        self._ratio = ratio
        self._test_size = int(math.floor(self._N * ratio))
        assert self._test_size > 0, 'ratio is too small.'
        self._train_size = self._N - self._test_size
        assert self._train_size > 0, 'ratio is too larger.'

    def split(self):
        shuffle_ix = np.random.permutation(self._N)
        train_ix = shuffle_ix[:self._train_size]
        test_ix = shuffle_ix[self._train_size:]
        return (train_ix, test_ix)
