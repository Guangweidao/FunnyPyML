# -*- coding: utf-8 -*-
from collections import defaultdict
import matplotlib.pyplot as plt


class FreqDict(object):
    def __init__(self, terms):
        assert type(terms) in [list, set], 'input must be list or set.'
        self.__fd = defaultdict(int)
        for term in terms:
            self.__fd[term] += 1

    def __getitem__(self, item):
        return self.__fd[item]

    def __iter__(self):
        return iter(self.__fd.iteritems())

    def print_freq_dict(self):
        for k, v in self.__fd.iteritems():
            print k, v

    def plot_pie(self):
        plt.pie(x=self.__fd.values(), labels=self.__fd.keys(), autopct='%1.1f%%', shadow=True)
        plt.show()


if __name__ == '__main__':
    testdata = [1, 2, 3, 1, 2, 5, 1, 9, 2, 5, 6, 2]
    fd = FreqDict(testdata)
    fd.plot_pie()
