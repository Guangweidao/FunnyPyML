# -*- coding: utf-8 -*-
import copy

import numpy as np

from base.common_function import euclidean_distance
from util.heap import Heap


class KDNode(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.axis = None
        self.father = None
        self.left_child = None
        self.right_child = None


class KDTree(object):
    def __init__(self, X, y, dist):
        super(KDTree, self).__init__()
        self.nSize = X.shape[0]
        self._nCol = X.shape[1]
        self._dist = dist
        self._search_count = 0
        self.root = self.build_kd_tree(X, y, 0)
        self.root.father = None

    def build_kd_tree(self, X, y, axis):
        y = y.reshape(-1, 1)
        axis = axis % self._nCol
        nSize = X.shape[0]
        _X = np.hstack((X, y))
        _X = np.array(sorted(_X, key=lambda d: d[axis]))
        _X, _y = _X[:, :-1], _X[:, -1]
        split_ix = nSize / 2
        node = KDNode()
        node.X, node.y, node.axis = _X[split_ix], _y[split_ix], axis
        if nSize != 1:
            L_X, L_y = _X[:split_ix, :], _y[:split_ix]
            if L_X.shape[0] > 0:
                node.left_child = self.build_kd_tree(L_X, L_y, axis + 1)
                node.left_child.father = node
            else:
                node.left_child = None
            R_X, R_y = _X[split_ix + 1:, :], _y[split_ix + 1:]
            if R_X.shape[0] > 0:
                node.right_child = self.build_kd_tree(R_X, R_y, axis + 1)
                node.right_child.father = node
            else:
                node.right_child = None
        else:
            node.left_child = None
            node.right_child = None
        return node

    def search(self, root, centriod, k):
        def neighbor_compare(a, b):
            return self._dist(a.X, centriod) > self._dist(b.X, centriod)

        candidate = Heap(neighbor_compare)
        nearest_neighbor = Heap(k, neighbor_compare)
        candidate.insert(root)
        self._search_count = 0
        while len(candidate) > 0:
            node = candidate.pop(1)
            is_search = True
            if len(nearest_neighbor) == k:
                father = node.father
                if father is not None:
                    _centriod = copy.deepcopy(centriod)
                    _centriod[father.axis] = father.X[father.axis]
                    if self._dist(_centriod, centriod) > self._dist(nearest_neighbor[1].X, centriod):
                        is_search = False
                    else:
                        is_search = True
            if is_search is True:
                self.search_subtree(node, centriod, candidate, nearest_neighbor)
        return nearest_neighbor

    def search_subtree(self, subtree, centroid, candidate, nearest_neighbor):
        if subtree is None:
            return
        nearest_neighbor.insert(subtree)
        self._search_count += 1
        if centroid[subtree.axis] > subtree.X[subtree.axis]:
            if subtree.left_child is not None:
                candidate.insert(subtree.left_child)
            if subtree.right_child is None:
                return
            else:
                self.search_subtree(subtree.right_child, centroid, candidate, nearest_neighbor)
        else:
            if subtree.right_child is not None:
                candidate.insert(subtree.right_child)
            if subtree.left_child is None:
                return
            else:
                return self.search_subtree(subtree.left_child, centroid, candidate, nearest_neighbor)

    def get_search_ratio(self):
        return float(self._search_count) / self.nSize

