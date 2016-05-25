# -*- coding: utf-8 -*-
import os
import numpy as np
from abstract_learner import AbstractClassifier
from util.freq_dict import FreqDict
from base.dataloader import DataLoader
from base.metric import accuracy_score
from abstract_learner import AbstractRegressor
from base.metric import mean_error
from util.kd_tree import KDTree
from base.common_function import euclidean_distance


class KNNClassifier(AbstractClassifier):
    def __init__(self, k=10, search_mode='kd_tree'):
        super(KNNClassifier, self).__init__()
        self._K = k
        self._search_mode = search_mode

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nDim = len(X.shape)
            self._nFeat = X.shape[1]
        if self._search_mode == 'kd_tree':
            self._parameter['kd_tree'] = KDTree(X, y, euclidean_distance)
            self._logger.info('KD Tree is builded up.')
        elif self._search_mode == 'brutal':
            self._parameter['neighbor_X'] = X
            self._parameter['neighbor_y'] = y
        else:
            raise ValueError
        self._is_trained = True

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        pred = list()
        if self._search_mode == 'kd_tree':
            kd_tree = self._parameter['kd_tree']
            K = min(self._K, self._parameter['kd_tree'].nSize)
            for i in xrange(X.shape[0]):
                neighbor = kd_tree.search(kd_tree.root, X[i, :], K)
                fd = FreqDict([v.y for v in neighbor], reverse=True)
                pred.append(fd.keys()[0])
                self._logger.info(
                    'progress : %.2f %%\tsearch ratio : %f' % (float(i) / X.shape[0] * 100, kd_tree.get_search_ratio()))
        elif self._search_mode == 'brutal':
            K = min(self._K, len(self._parameter['neighbor_y']))
            for i in xrange(X.shape[0]):
                dist = list()
                for irow in range(self._parameter['neighbor_X'].shape[0]):
                    dist.append(np.linalg.norm(X[i, :] - self._parameter['neighbor_X'][irow, :]))
                indices = np.argsort(dist)[:K]
                fd = FreqDict(list(self._parameter['neighbor_y'][indices]), reverse=True)
                pred.append(fd.keys()[0])
                self._logger.info('progress: %.2f %%' % (float(i) / X.shape[0] * 100))
        else:
            raise ValueError
        return pred

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nDim = len(X.shape)
            if nFeat == self._nFeat and nDim == self._nDim:
                is_valid = True
            return is_valid


class KNNRegressor(AbstractRegressor):
    def __init__(self, k=10, search_mode='kd_tree'):
        super(KNNRegressor, self).__init__()
        self._K = k
        self._search_mode = search_mode

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nDim = len(X.shape)
            self._nFeat = X.shape[1]
        if self._search_mode == 'kd_tree':
            self._parameter['kd_tree'] = KDTree(X, y, euclidean_distance)
            self._logger.info('KD Tree is builded up.')
        elif self._search_mode == 'brutal':
            self._parameter['neighbor_X'] = X
            self._parameter['neighbor_y'] = y
        else:
            raise ValueError
        self._is_trained = True

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nDim = len(X.shape)
            if nFeat == self._nFeat and nDim == self._nDim:
                is_valid = True
            return is_valid

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        pred = list()
        if self._search_mode == 'kd_tree':
            kd_tree = self._parameter['kd_tree']
            K = min(self._K, self._parameter['kd_tree'].nSize)
            for i in xrange(X.shape[0]):
                neighbor = kd_tree.search(kd_tree.root, X[i, :], K)
                pred.append(np.mean([node.y for node in neighbor]))
                self._logger.info(
                    'progress : %.2f %%\tsearch ratio : %f' % (float(i) / X.shape[0] * 100, kd_tree.get_search_ratio()))
        elif self._search_mode == 'brutal':
            K = min(self._K, len(self._parameter['neighbor_y']))
            for i in xrange(X.shape[0]):
                dist = list()
                for irow in range(self._parameter['neighbor_X'].shape[0]):
                    dist.append(np.linalg.norm(X[i, :] - self._parameter['neighbor_X'][irow, :]))
                indices = np.argsort(dist)[:K]
                pred.append(np.mean(self._parameter['neighbor_y'][indices]))
                self._logger.info('progress: %.2f %%' % (float(i) / X.shape[0] * 100))
        else:
            raise ValueError
        return pred


if __name__ == '__main__':
    from base.time_scheduler import TimeScheduler

    scheduler = TimeScheduler()

    # KNN for classification task
    path = os.getcwd() + '/../dataset/electricity-normalized.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='class')
    trainset, testset = dataset.cross_split()
    knn = KNNClassifier(search_mode='kd_tree')
    knn.fit(trainset[0], trainset[1])
    predict_kd_tree = scheduler.tic_tac('kd_tree', knn.predict, X=testset[0])
    knn = KNNClassifier(search_mode='brutal')
    knn.fit(trainset[0], trainset[1])
    predict_brutal = scheduler.tic_tac('brutal', knn.predict, X=testset[0])
    scheduler.print_task_schedule('brutal')
    scheduler.print_task_schedule('kd_tree')
    print accuracy_score(testset[1], predict_brutal), accuracy_score(testset[1], predict_kd_tree)

    # KNN for regression task
    # path = os.getcwd() + '/../dataset/winequality-white.csv'
    # loader = DataLoader(path)
    # dataset = loader.load(target_col_name='quality')
    # trainset, testset = dataset.cross_split()
    # knn = KNNRegressor(search_mode='brutal')
    # knn.fit(trainset[0], trainset[1])
    # predict_brutal = scheduler.tic_tac('brutal', knn.predict, X=testset[0])
    # knn = KNNRegressor(search_mode='kd_tree')
    # knn.fit(trainset[0], trainset[1])
    # predict_kd_tree = scheduler.tic_tac('kd_tree', knn.predict, X=testset[0])
    # scheduler.print_task_schedule('brutal')
    # scheduler.print_task_schedule('kd_tree')
    # print mean_error(testset[1], predict_brutal), mean_error(testset[1], predict_kd_tree)
