# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from abstract_learner import AbstractClassifier, AbstractRegressor
from base.dataloader import DataLoader
from base.common_function import entropy, condition_entropy
from util.freq_dict import FreqDict
from base.metric import accuracy_score
from util.shuffle_split import ShuffleSpliter
from base._logging import get_logger

logger = get_logger(__name__)


class DecisionTreeClassifier(AbstractClassifier):
    def __init__(self, min_split=5, is_prune=True):
        super(DecisionTreeClassifier, self).__init__()
        self._min_split = min_split
        self._is_prune = is_prune

    def fit(self, X, y):
        assert self.__check_valid(X, y), 'input is invalid.'
        if self._is_trained is False:
            self._nFeat = X.shape[1]
            self._nClass = len(np.unique(y))
        if self._is_prune is True:
            spliter = ShuffleSpliter(X.shape[0], ratio=0.3)
            train_ix, val_ix = spliter.split()
            Xtrain, ytrain = X[train_ix], y[train_ix]
            Xval, yval = X[val_ix], y[val_ix]
            used_feat = set()
            self._parameter['tree'] = self._build_tree(Xtrain, ytrain, used_feat)
            self._tree_prune(self._parameter['tree'], Xval, yval)
        else:
            used_feat = set()
            self._parameter['tree'] = self._build_tree(X, y, used_feat)
        self._is_trained = True

    def _tree_prune(self, tree, Xval, yval):
        feat = tree.keys()[0]
        for feat_val in tree[feat].keys():
            if isinstance(tree[feat][feat_val], dict):
                subtree = tree[feat][feat_val]
                pred_no_prone = self.predict(Xval)
                perf_no_prone = accuracy_score(yval, pred_no_prone)
                num_leaf_no_prone = self.get_num_leafs(self._parameter['tree'])
                tree[feat][feat_val] = subtree[subtree.keys()[0]]['__default__']
                pred_with_prone = self.predict(Xval)
                perf_with_prone = accuracy_score(yval, pred_with_prone)
                num_leaf_with_prone = self.get_num_leafs(self._parameter['tree'])
                if perf_no_prone < perf_with_prone:
                    improve = perf_with_prone - perf_no_prone
                    # improve /= num_leaf_no_prone - num_leaf_with_prone - 1
                    logger.info('tree prune, validation precision improve %f' % (improve))
                else:
                    tree[feat][feat_val] = subtree
                    self._tree_prune(subtree, Xval, yval)

    def predict(self, X):
        return self._predict(X, self._parameter['tree'])

    def _predict(self, X, tree):
        nSize = X.shape[0]
        pred = list()
        for irow in xrange(nSize):
            _x = X[irow]
            pred.append(self._classify(_x, tree))
        return np.array(pred)

    def _classify(self, x, subtree):
        if isinstance(subtree, dict):
            feat = subtree.keys()[0]
            feat_val = x[feat]
            if feat_val not in subtree[feat]:
                return subtree[feat]['__default__']
            else:
                return self._classify(x, subtree[feat][feat_val])
        else:
            return subtree

    def _build_tree(self, X, y, used_feat):
        if len(np.unique(y)) == 1:
            return y[0]
        if X.shape[1] == 1:
            return FreqDict(list(y), reverse=True).keys()[0]
        if len(y) < self._min_split:
            return FreqDict(list(y), reverse=True).keys()[0]
        _used_feat = copy.deepcopy(used_feat)
        choosed_feat = self._choose_feature(X, y, _used_feat)
        _used_feat.add(choosed_feat)
        root = {choosed_feat: {}}
        root[choosed_feat]['__default__'] = FreqDict(list(y), reverse=True).keys()[0]
        for v in np.unique(X[:, choosed_feat]):
            indices = X[:, choosed_feat] == v
            root[choosed_feat][v] = self._build_tree(X[indices], y[indices], _used_feat)
        return root

    def _choose_feature(self, X, y, used_feat):
        info_init = entropy(y)
        max_gain = None
        choosed_feat = None
        for i in xrange(X.shape[1]):
            if i in used_feat:
                continue
            info_split = condition_entropy(X[:, i], y)
            gain = info_init - info_split
            if gain > max_gain or max_gain is None:
                max_gain = gain
                choosed_feat = i
        return choosed_feat

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

    def get_num_leafs(self, subtree):
        num_leafs = 0
        feat = subtree.keys()[0]
        for k, v in subtree[feat].iteritems():
            if isinstance(v, dict):
                num_leafs += self.get_num_leafs(v)
            else:
                num_leafs += 1
        return num_leafs

    def get_tree_depth(self, subtree):
        max_depth = 0
        feat = subtree.keys()[0]
        for k, v in subtree[feat].iteritems():
            depth = 1
            if isinstance(v, dict):
                depth += self.get_tree_depth(v)
            if depth > max_depth:
                max_depth = depth
        return max_depth

    def plotNode(self, nodeTxt, centerPt, parentPt, nodeType):
        self.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                          va='center', ha='center', bbox=nodeType, arrowprops=self.arrow_args)

    def plotMidText(self, cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        self.ax1.text(xMid, yMid, txtString)

    def plotTree(self, myTree, parentPt, nodeTxt):
        numLeafs = self.get_num_leafs(myTree)
        depth = self.get_tree_depth(myTree)
        firstStr = myTree.keys()[0]
        cntrPt = (self.xOff + (1.0 + float(numLeafs)) / 2.0 / self.totalW, self.yOff)
        self.plotMidText(cntrPt, parentPt, nodeTxt)
        self.plotNode(firstStr, cntrPt, parentPt, self.decisionNode)
        secondDict = myTree[firstStr]
        self.yOff -= 1.0 / self.totalD
        for key in secondDict.keys():
            if type(secondDict[key]).__name__ == 'dict':
                self.plotTree(secondDict[key], cntrPt, str(key))
            else:
                self.xOff += 1.0 / self.totalW
                self.plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, self.leafNone)
                self.plotMidText((self.xOff, self.yOff,), cntrPt, str(key))
        self.yOff += 1.0 / self.totalD

    def createPlot(self):
        inTree = self._parameter['tree']
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = dict(xticks=[], yticks=[])
        self.decisionNode = dict(boxstyle='sawtooth', fc='0.8')
        self.leafNone = dict(boxstyle='round4', fc='0.8')
        self.arrow_args = dict(arrowstyle='<-')
        self.ax1 = plt.subplot(111, frameon=False, **axprops)
        self.totalW = float(self.get_num_leafs(inTree))
        self.totalD = float(self.get_tree_depth(inTree))
        self.xOff = -0.5 / self.totalW
        self.yOff = 1.0
        self.plotTree(inTree, (0.5, 1.0), '')
        plt.show()


if __name__ == '__main__':
    path = os.getcwd() + '/../dataset/dataset_21_car.arff'
    loader = DataLoader(path)
    dataset = loader.load(target_col_name='class')
    trainset, testset = dataset.cross_split()
    dt = DecisionTreeClassifier(min_split=3, is_prune=True)
    dt.fit(trainset[0], trainset[1])
    dt.createPlot()
    predict = dt.predict(testset[0])
    performance = accuracy_score(testset[1], predict)
    print 'test accuracy:', performance
