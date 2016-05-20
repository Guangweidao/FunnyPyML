# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
import copy


class Dataset(object):
    def __init__(self, data, target_col_name, categorical):
        assert isinstance(data, pd.DataFrame), 'input must be a dataframe.'
        assert target_col_name in data.columns, 'target column name does not existed.'
        self.__data = data
        self.__target_col_name = target_col_name
        self.__categorical = categorical
        self.__nSize = data.shape[0]
        self.__nFeat = data.shape[1]

    def cross_split(self, ratio=0.1, shuffle=True):
        assert self.__nSize > ratio * 10, 'ratio is too large.'
        test_size = math.floor(self.__nSize * ratio)
        assert test_size < 1, 'ratio is too small so that no testset will be split.'
        train_size = self.__nSize - test_size
        X = copy.deepcopy(self.__data)
        y = X.pop(self.__target_col_name)
        X = X.as_matrix()
        y = y.as_matrix()
        if shuffle is True:
            shuffle_ix = np.random.shuffle(range(self.__nSize))
            X = X[shuffle_ix]
            y = y[shuffle_ix]
        trainset = (X[:train_size], y[:train_size])
        testset = (X[train_size:], y[train_size:])
        return trainset, testset
