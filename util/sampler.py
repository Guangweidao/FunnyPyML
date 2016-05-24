# -*- coding: utf-8 -*-
import math
import numpy as np


class Sampler(object):
    def __init__(self, n, k, mode='fast'):
        self._N = int(n)
        self._K = int(k)
        self._mode = mode

    def sample(self, sample_weight=None, replacement=True):
        if sample_weight is not None:
            assert type(sample_weight) in [list, np.ndarray], 'sample weight must be a list.'
            assert len(sample_weight) == self._N, 'sample weight number must equal to %d.' % (self._N)
            sample_weight = list(sample_weight)
        else:
            sample_weight = [1. / self._N] * self._N
        if replacement is False:
            return np.random.permutation(self._N)[:self._K]
        if self._mode == 'fast':
            return self._fast_sample_by_proba(sample_weight)
        elif self._mode == 'normal':
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
        start_ix = 0
        end_ix = 0
        for i in xrange(self._N):
            if i == self._N:
                cdf_map.extend([i] * (self._N - start_ix + 1))
                break
            end_ix += int(math.floor(sample_weight[i] * nArray))
            cdf_map.extend([i] * (end_ix - start_ix + 1))
            start_ix = end_ix
        cdf_map = np.array(cdf_map)
        indices = np.random.randint(0, nArray, self._K)
        return cdf_map[indices]


if __name__ == '__main__':
    from base.time_scheduler import TimeScheduler

    scheduler = TimeScheduler()
    fast_sampler = Sampler(1e5, 1e3, mode='fast')
    normal_sampler = Sampler(1e5, 1e3, mode='normal')
    scheduler.tic_tac('sample_fast', fast_sampler.sample)
    scheduler.tic_tac('sample_normal', normal_sampler.sample)
    scheduler.print_task_schedule('sample_fast')
    scheduler.print_task_schedule('sample_normal')
