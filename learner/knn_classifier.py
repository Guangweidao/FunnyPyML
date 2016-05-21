# -*- coding: utf-8 -*-
from abstract_learner import AbstractClassifier

class KNNClassifier(AbstractClassifier):
    def fit(self, X, y):
        if self._is_trained is False:
            pass
