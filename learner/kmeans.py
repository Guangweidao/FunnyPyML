# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from abstract_learner import AbstractCluster
from base.common_function import euclidean_distance
from base.dataloader import DataLoader
from base.metric import cluster_f_measure
from base._logging import get_logger


class Kmeans(AbstractCluster):
    def __init__(self, k=2, is_plot=True):
        super(Kmeans, self).__init__()
        self._K = k
        self._is_plot = is_plot

    def predict(self, X):
        assert self._is_trained, 'model must be trained before predict.'
        assert self.__check_valid(X), 'input is invalid.'
        nSize = X.shape[0]
        cluster_belong = list()
        for irow in range(nSize):
            min_dist = None
            min_centroid = None
            for ix, v in enumerate(self._parameter['cluster_center']):
                dist = euclidean_distance(X[irow], v)
                if dist < min_dist or min_dist is None:
                    min_dist = dist
                    min_centroid = ix
            cluster_belong.append(min_centroid)
        return np.array(cluster_belong)

    def fit(self, X):
        assert self.__check_valid(X), 'input is invalid.'
        if self._is_trained is False:
            self._nFeat = X.shape[1]
        nSize = X.shape[0]
        init_cluster_indices = np.random.permutation(nSize)[:self._K]
        centroids = [X[ix] for ix in init_cluster_indices]
        last_total_dist = None
        while True:
            total_dist = 0
            cluster_belong = {i: list() for i in range(len(centroids))}
            for irow in range(nSize):
                min_dist = None
                min_centroid = None
                for k in cluster_belong.keys():
                    dist = euclidean_distance(centroids[k], X[irow])
                    if dist < min_dist or min_dist is None:
                        min_dist = dist
                        min_centroid = k
                cluster_belong[min_centroid].append(irow)
                total_dist += min_dist
            for k, v in cluster_belong.iteritems():
                centroids[k] = np.mean(np.array(X[v]), axis=0)
            if total_dist == last_total_dist:
                break
            else:
                last_total_dist = total_dist
            logger.info('total distance: %f' % (total_dist))
        self._parameter['cluster_center'] = centroids
        if self._is_plot is True:
            self.plot_scatter(X, centroids, cluster_belong)
        self._is_trained = True

    def plot_scatter(self, X, centroids, cluster_belong):
        nFeat = X.shape[1]
        if nFeat != 2:
            logger.warning('feature number must be 2.')
            return
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if self._K > len(mark):
            logger.warning('k is too large.')
            return
        # plot all point
        for i in cluster_belong.keys():
            for ix in cluster_belong[i]:
                plt.plot(X[ix, 0], X[ix, 1], mark[i])
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # plot centroids
        for i, centroid in enumerate(centroids):
            plt.plot(centroid[0], centroid[1], mark[i], markersize=12)
        plt.show()

    def __check_valid(self, X):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            if nFeat == self._nFeat:
                is_valid = True
            return is_valid


logger = get_logger(Kmeans.__name__)

if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/iris.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='binaryClass')
    trainset, testset = dataset.cross_split()
    kmeans = Kmeans(2, is_plot=True)
    kmeans.fit(trainset[0][:, [1, 3]])
    prediction = kmeans.predict(testset[0][:, [1, 3]])
    performance = cluster_f_measure(testset[1], prediction)
    print 'F-measure:', performance
