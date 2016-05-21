# -*- coding: utf-8 -*-

import os
import numpy as np
from abstract_learner import AbstractClassifier
from base.dataloader import DataLoader
from util.freq_dict import FreqDict
from collections import defaultdict
from base.metric import accuracy_score


class NaiveBayes(AbstractClassifier):
    def __init__(self):
        super(NaiveBayes, self).__init__()

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nFeat = X.shape[1]
            self._nClass = len(np.unique(y))
            self.feat_set = dict()
            for icol in range(X.shape[1]):
                self.feat_set[icol] = list(np.unique(X[:, icol]))
        freq_y = {k: v for k, v in FreqDict(list(y))}
        proba_y = {k: np.log(float(v) / len(y)) for k, v in freq_y.iteritems()}
        cond_proba_feat = {k: {i: defaultdict(float) for i in range(X.shape[1])} for k in proba_y.keys()}
        for c in proba_y.keys():
            for icol in self.feat_set.keys():
                for feat_val in self.feat_set[icol]:
                    cond_proba_feat[c][icol][feat_val] = 1
        for irow in range(X.shape[0]):
            for icol in range(X.shape[1]):
                cond_proba_feat[y[irow]][icol][X[irow, icol]] += 1
        for c, feats in cond_proba_feat.iteritems():
            for icol, feat in feats.iteritems():
                for feat_val in feat.keys():
                    feat[feat_val] = np.log(float(feat[feat_val]) / (freq_y[c] + len(self.feat_set[icol])))
        self._parameter['proba_y'] = proba_y
        self._parameter['cond_proba_feat'] = cond_proba_feat

    def __check_valid(self, X, y):
        if self._is_trained is False:
            return True
        else:
            is_valid = False
            nFeat = X.shape[1]
            nClass = len(np.unique(y))
            if nFeat == self._nFeat and nClass == self._nClass:
                is_valid = True
            for icol in range(X.shape[1]):
                for irow in range(X.shape[0]):
                    if not X[irow, icol] in self.feat_set[icol]:
                        is_valid = False
            return is_valid

    def predict(self, X):
        proba_y = self._parameter['proba_y']
        cond_proba_y = self._parameter['cond_proba_feat']
        pred = list()
        for irow in range(X.shape[0]):
            _X = X[irow]
            max_prob = None
            label = None
            for c in proba_y.keys():
                p = proba_y[c]
                for icol, feat in cond_proba_y[c].iteritems():
                    p += feat[_X[icol]]
                if max_prob < p or max_prob is None:
                    max_prob = p
                    label = c
            assert label is not None, 'label should be None. There must be some error. please check.'
            pred.append(label)
        return np.array(pred)


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/dataset_21_car.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='Class')
    trainset, testset = dataset.cross_split()
    nb = NaiveBayes()
    nb.fit(trainset[0], trainset[1])
    predict = nb.predict(testset[0])
    acc = accuracy_score(testset[1], predict)
    print acc
    nb.dump('NB.model')
    # nb = NaiveBayes.load('NB.model')
    # predict = nb.predict(testset[0])
    # print accuracy_score(testset[1], predict)
