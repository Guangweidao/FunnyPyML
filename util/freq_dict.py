# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
import matplotlib.pyplot as plt
import numpy as np

class FreqDict(object):
    def __init__(self, terms, reverse=False):
        assert type(terms) in [list, set, np.ndarray], 'input must be list, set or ndarray.'
        fd = defaultdict(int)
        for term in terms:
            fd[term] += 1
        self.__fd = OrderedDict()
        for k, v in sorted(fd.iteritems(), key=lambda d: d[1], reverse=reverse):
            self.__fd[k] = v

    def __getitem__(self, item):
        return self.__fd[item]

    def __iter__(self):
        return iter(self.__fd.iteritems())

    def keys(self):
        return self.__fd.keys()

    def values(self):
        return self.__fd.values()

    def print_freq_dict(self):
        for k, v in self.__fd.iteritems():
            print k, v

    def plot_pie(self):
        plt.pie(x=self.__fd.values(), labels=self.__fd.keys(), autopct='%1.1f%%', shadow=True)
        plt.show()


if __name__ == '__main__':
    testdata = [1, 2, 3, 1, 2, 5, 1, 9, 2, 5, 6, 2]
    fd = FreqDict(testdata)
    fd.print_freq_dict()
    print fd.keys()
