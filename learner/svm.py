# -*- coding: utf-8 -*-
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from abstract_learner import AbstractClassifier
from base.dataloader import DataLoader
from base.metric import accuracy_score


class SVM(AbstractClassifier):
    def __init__(self, max_iter, kernel_type='linear', tol=1e-1, C=1, **kwargs):
        super(SVM, self).__init__()
        self._max_iter = max_iter
        self._kernel_type = kernel_type
        self._label2ix = dict()
        self._ix2label = dict()
        self._nFeat = None
        self._nClass = None
        self._tol = tol
        self._C = C
        self._sigma = None
        self._p = None
        if self._kernel_type == 'rbf':
            if 'sigma' not in kwargs:
                self._logger.info(
                    'kernel type is rbf (gaussian kernel), but you dont indicate sigma, default sigma=0.5')
                self._sigma = 0.5
            else:
                self._sigma = kwargs['sigma']
        elif self._kernel_type == 'poly':
            if 'p' not in kwargs:
                self._logger.info(
                    'kernel type is polynomial, but you dont indicate p, default p=1')
                self._p = 1
            else:
                self._p = kwargs['p']

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        nSize, nFeat = X.shape[0], X.shape[1]
        nClass = len(np.unique(y))
        if self._is_trained is False:
            self._nFeat = nFeat
            self._nClass = nClass
        b = np.zeros(1)
        alpha = np.zeros((nSize, 1)).astype(np.float64)
        self._label2ix = {v: i for i, v in enumerate(np.unique(y))}
        self._ix2label = {i: v for v, i in self._label2ix.iteritems()}
        y = np.array([-1 if self._label2ix[v] == 0 else 1 for v in y])
        y = np.reshape(y, (-1, 1)).astype(np.float64)
        K = np.zeros((nSize, nSize))
        for i in xrange(nSize):
            K[:, i] = self._kernel_transform(X, X[i])

        for i in xrange(self._max_iter):
            alpha, b = self._platt_smo(K, y, alpha, b)

        indices = np.nonzero(alpha)[0]
        self._parameter['alpha'] = alpha[indices]
        self._parameter['support_vector'] = X[indices, :]
        self._parameter['y'] = y[indices]
        self._parameter['b'] = b
        self._is_trained = True

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nClass = len(np.unique(y))
            if nFeat == self._nFeat and nClass == self._nClass:
                is_valid = True
            return is_valid

    def _platt_smo(self, K, y, alpha, b):
        nSize = len(y)
        sv = set(np.nonzero(alpha)[0])  # support vector
        C = self._C
        for ix1 in xrange(nSize):
            e = self._update_e(K, y, alpha, b)
            usv = set(range(nSize)) - sv
            e1 = e[ix1]
            if ((y[ix1] * e1 < -self._tol) and (alpha[ix1] < C)) or ((y[ix1] * e1 > self._tol) and (alpha[ix1] > 0)):
                ix2 = self._select_alpha(ix1, e, sv, usv)
                alpha, b = self._update_alpha_b(K, y, alpha, b, ix1, ix2, e)
                if 0 < alpha[ix1] < C:
                    sv.add(ix1)
                if 0 < alpha[ix2] < C:
                    sv.add(ix2)
        return alpha, b

    def _select_alpha(self, ix1, e, sv, usv):
        e1 = e[ix1]
        max_de = -1
        max_ix2 = -1
        for i in sv:
            if ix1 == i:
                continue
            e2 = e[i]
            de = abs(e1 - e2)
            if de > max_de:
                max_de = de
                max_ix2 = i
        if max_de > 0:
            return max_ix2

        for i in usv:
            if ix1 == i:
                continue
            e2 = e[i]
            de = abs(e1 - e2)
            if de > max_de:
                max_de = de
                max_ix2 = i
        return max_ix2

    def _update_e(self, K, y, alpha, b):
        nSize = len(y)
        e = list()
        for i in xrange(nSize):
            fxi = np.dot(K[i], alpha * y) + b
            ei = fxi - y[i]
            e.append(ei)
        return e

    def _update_alpha_b(self, K, y, alpha, b, ix1, ix2, e):
        C = self._C
        e1 = e[ix1]
        e2 = e[ix2]

        if y[ix1] != y[ix2]:
            L = max(0, alpha[ix2] - alpha[ix2])
            H = min(C, C + alpha[ix2] - alpha[ix1])
        else:
            L = max(0, alpha[ix1] + alpha[ix2] - C)
            H = min(C, alpha[ix1] + alpha[ix2])
        eta = 2. * K[ix1, ix2] - K[ix1, ix1] - K[ix2, ix2]
        if eta >= 0:
            return alpha, b

        # update alpha
        alpha1_old = copy.deepcopy(alpha[ix1])
        alpha2_old = copy.deepcopy(alpha[ix2])
        alpha2_unclip = alpha2_old - y[ix2] * (e1 - e2) / eta
        alpha[ix2] = self._clip_alpha(alpha2_unclip, L, H)
        alpha[ix1] = alpha1_old + y[ix1] * y[ix2] * (alpha2_old - alpha[ix2])

        # update b
        b1 = -e1 - y[ix1] * K[ix1, ix1] * (alpha[ix1] - alpha1_old) - y[ix2] * K[ix2, ix1] * (
            alpha[ix2] - alpha2_old) + b
        b2 = -e2 - y[ix1] * K[ix1, ix2] * (alpha[ix1] - alpha1_old) - y[ix2] * K[ix2, ix2] * (
            alpha[ix2] - alpha2_old) + b

        if 0 < alpha[ix1] < C:
            b = b1
        elif 0 < alpha[ix2] < C:
            b = b2
        else:
            b = (b1 + b2) / 2.
        return alpha, b

    def _kernel_transform(self, x1, x2):
        if self._kernel_type == 'linear':
            return np.dot(x1, x2.T)
        elif self._kernel_type == 'poly':
            p = self._p if self._p is not None else 1
            return (np.dot(x1, x2.T) + 1) ** p
        elif self._kernel_type == 'rbf':
            sigma = self._sigma if self._sigma is not None else 0.5
            nSize = x1.shape[0]
            _k = np.zeros(nSize)
            for i in xrange(nSize):
                dx = x1[i] - x2
                _k[i] = np.dot(dx, dx.T)
            return np.exp(_k / (-2. * sigma ** 2))
        else:
            raise ValueError

    def _clip_alpha(self, alpha, L, H):
        if alpha > H:
            return H
        elif alpha < L:
            return L
        else:
            return alpha

    def predict(self, X):
        pred = self._predict(X)
        pred = [1 if p == 1 else 0 for p in pred]
        return np.array([self._ix2label[v] for v in pred])

    def _predict(self, X):
        assert self._is_trained, 'model should be fitted.'
        nSize = X.shape[0]
        pred = np.empty(nSize)
        sv, y = self._parameter['support_vector'], self._parameter['y']
        alpha, b = self._parameter['alpha'], self._parameter['b']
        for i in xrange(nSize):
            p = np.dot((alpha * y).T, self._kernel_transform(sv, X[i])) + b
            pred[i] = np.sign(p)
        return pred

    def plot(self, X):
        nSize, nFeat = X.shape
        if nFeat != 2:
            self._logger.warning('feature number must be 2.')
            return
        self._logger.info('start plotting...')
        pred = self._predict(X)
        h = 0.02  # step size in the mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = self._predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=pred, cmap=plt.cm.Paired)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
        plt.show()


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/iris.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='binaryClass')
    trainset, testset = dataset.cross_split()
    X = trainset[0][:, [0, 1]]
    y = trainset[1]
    svm = SVM(5, kernel_type='rbf', sigma=0.3)
    svm.fit(X, y)
    predict = svm.predict(testset[0][:, [0, 1]])
    print 'test accuracy:', accuracy_score(testset[1], predict)
    svm.plot(X)
