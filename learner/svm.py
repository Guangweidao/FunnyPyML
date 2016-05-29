# -*- coding: utf-8 -*-
import os
import numpy as np
from abstract_learner import AbstractClassifier
from base.dataloader import DataLoader
from base.metric import accuracy_score


class SVM(AbstractClassifier):
    def __init__(self, max_iter):
        super(SVM, self).__init__()
        self._max_iter = max_iter
        self._label2ix = dict()
        self._ix2label = dict()
        self._w = None
        self._b = None
        self._tol = 1e-3
        self._C = 0.6

    def fit(self, X, y):
        nSize, nFeat = X.shape[0], X.shape[1]
        w, b = np.zeros((nFeat, 1)), 0
        alpha = np.zeros((nSize, 1)).astype(np.float64)
        self._label2ix = {v: i for i, v in enumerate(np.unique(y))}
        self._ix2label = {i: v for v, i in self._label2ix.iteritems()}
        y = np.array([-1 if self._label2ix[v] == 0 else 1 for v in y])
        y = np.reshape(y, (-1, 1)).astype(np.float64)
        epoch = 0
        while epoch < self._max_iter:
            alphaPairsChanged = 0
            for i in xrange(nSize):
                fxi = np.dot((alpha * y).T, np.dot(X, X[i, :].T)) + b
                Ei = fxi - y[i]
                if ((y[i] * Ei < -self._tol) and (alpha[i] < self._C)) or ((y[i] * Ei > self._tol) and (alpha[i] > 0)):
                    j = self.selectJrand(i, nSize)
                    fxj = np.dot((alpha * y).T, np.dot(X, X[j, :].T)) + b
                    Ej = fxj - y[j]
                    alphaIold = alpha[i].copy()
                    alphaJold = alpha[j].copy()
                    if y[i] != y[j]:
                        L = max([0, alpha[j] - alpha[i]])
                        H = min([self._C, self._C + alpha[j] - alpha[i]])
                    else:
                        L = max([0, alpha[j] + alpha[i] - self._C])
                        H = min([self._C, alpha[j] + alpha[i]])
                    if L == H:
                        print 'L==H'
                        continue
                    eta = 2.0 * np.dot(X[i, :], X[j, :].T) - np.dot(X[i, :], X[i, :].T) - np.dot(X[j, :], X[j, :].T)
                    if eta >= 0:
                        print 'eta >= 0'
                        continue
                    alpha[j] -= y[j] * (Ei - Ej) / eta
                    alpha[j] = self.clipAlpha(alpha[j], H, L)
                    if abs(alpha[j] - alphaJold) < 0.00001:
                        print 'j not moving enough'
                        continue
                    alpha[i] += y[j] * y[i] * (alphaJold - alpha[j])
                    b1 = b - Ei - y[i] * (alpha[i] - alphaIold) * np.dot(X[i, :], X[i, :].T) - y[j] * (
                        alpha[j] - alphaJold) * np.dot(X[i, :], X[j, :].T)
                    b2 = b - Ej - y[i] * (alpha[i] - alphaIold) * np.dot(X[i, :], X[j, :].T) - y[j] * (
                        alpha[j] - alphaJold) * np.dot(X[j, :], X[j, :].T)
                    if (0 < alpha[i]) and (self._C > alpha[i]):
                        b = b1
                    elif (0 < alpha[j]) and (self._C > alpha[j]):
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alphaPairsChanged += 1
                    print 'epoch:%d  i:%d, pairs changed %d' % (epoch, i, alphaPairsChanged)
            if alphaPairsChanged == 0:
                epoch += 1
            else:
                epoch = 0
            print 'epoch number:%d' % epoch

        for i in xrange(nSize):
            w += np.reshape(alpha[i] * y[i] * X[i, :].T, (-1, 1))
        self._w = w
        self._b = b

    def selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def predict(self, X):
        nSize = X.shape[0]
        pred = list()
        for i in xrange(nSize):
            pred.append(1 if np.dot(X[i, :], self._w) + self._b > 0 else 0)
        return np.array([self._ix2label[v] for v in pred])


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/iris.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='binaryClass')
    trainset, testset = dataset.cross_split()
    X = trainset[0][:, [0, 2]]
    y = trainset[1]
    svm = SVM(100)
    svm.fit(X, y)
    predict = svm.predict(testset[0][:, [0, 2]])
    print accuracy_score(testset[1], predict)
