# -*- coding: utf-8 -*-
import os
import arff
import pandas as pd
import numpy as np
from dataset import Dataset


class DataLoader(object):
    def __init__(self, path, data_type='auto'):
        assert self.__check_type(data_type), 'data type does not support.'
        assert os.path.exists(path), 'path does not exist.'
        self.__path = path
        self.__data_type = data_type
        self.__loader = self.__get_loader()

    def load(self, target_col_name=None):
        with open(self.__path, 'r') as fp:
            data, categorical = self.__loader(fp)
            target_col_name = data.columns[
                -1] if target_col_name is None or target_col_name not in data.columns else target_col_name
        return Dataset(data=data, target_col_name=target_col_name, categorical=categorical)

    def __load_arff(self, fp):
        arff_data = arff.load(fp)
        attr_name = [attr_info[0] for attr_info in arff_data['attributes']]
        value = arff_data['data']
        data = pd.DataFrame(data=value, columns=attr_name)
        categorical = self.__categorize_data(arff_data)
        return data, categorical

    def __load_csv(self, fp):
        data = pd.read_csv(fp)
        categorical = self.__categorize_data(data)
        return data, categorical

    def __get_loader(self):
        if self.__data_type == 'arff':
            return self.__load_arff
        elif self.__data_type == 'csv':
            return self.__load_csv
        elif self.__data_type == 'auto':
            if self.__path.endswith('.arff'):
                self.__data_type = 'arff'
                return self.__load_arff
            elif self.__path.endswith('.csv'):
                self.__data_type = '.csv'
                return self.__load_csv
        else:
            raise ValueError

    def __check_type(self, data_type):
        if data_type in ['arff', 'csv', 'auto']:
            return True
        else:
            return False

    def __categorize_data(self, data):
        categorical = pd.Series()
        if self.__data_type == 'arff':
            assert isinstance(data, dict), 'data must be a dict.'
            for attr_info in data['attributes']:
                attr_name = attr_info[0]
                attr_type = attr_info[1]
                if type(attr_type) in [list, set]:
                    categorical[attr_name] = True
                else:
                    categorical[attr_name] = False
        else:
            assert isinstance(data, pd.DataFrame), 'data must be a dataframe.'
            nRow = data.shape[0]
            for k, v in data.iteritems():
                if v.dtype == 'object' or len(np.unique(v)) < min([20, nRow / 10]):
                    categorical[k] = True
                else:
                    categorical[k] = False
