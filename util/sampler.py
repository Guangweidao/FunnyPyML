# -*- coding: utf-8 -*-
import math
import numpy as np


class Sampler(object):
    def __init__(self, n, k):
        self._N = n
        self._K = k

    def sample(self, sample_weight=None, mode='fast'):
        if sample_weight is not None:
            assert type(sample_weight) in [list, np.ndarray], 'sample weight must be a list.'
            assert len(sample_weight) == self._N, 'sample weight number must equal to %d.' % (self._N)
            sample_weight = list(sample_weight)
        else:
            sample_weight = [1. / self._N] * self._N
        if mode == 'fast':
            return self._fast_sample_by_proba(sample_weight)
        elif mode == 'normal':
            return self._sample_by_proba(sample_weight)
        else:
            raise ValueError

    def _sample_by_proba(self, sample_weight):
        cdf = list()
        total = 0.
        for i in xrange(self._N):
            total += sample_weight[i]
            cdf.append(total)
        result = list()
        for i in xrange(self._K):
            num = np.random.rand()
            for ix, v in enumerate(cdf):
                if v < num:
                    continue
                else:
                    result.append(ix)
                    break
        return np.array(result)

    def _fast_sample_by_proba(self, sample_weight):
        nArray = 1e5
        cdf_map = list()
        for i in xrange(self._N):
            m = int(math.floor(sample_weight[i] * nArray))
            cdf_map.extend([i] * m)
        indices = np.random.randint(0, nArray, self._K)
        return cdf_map[indices]





