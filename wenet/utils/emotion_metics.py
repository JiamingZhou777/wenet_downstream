import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb

'''
https://github.com/b04901014/FT-w2v2-ser/blob/e70175c2620b786269105ab0301cc0caab2911d0/utils/metrics.py#L54
'''
class ConfusionMetrics:
    def __init__(self, n_classes):
        self.nc = n_classes
        self.m = np.zeros([n_classes, n_classes], dtype=np.float32)

    def __add__(self, other):
        self.m += other.m
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def fit(self, target, pred):
        np.add.at(self.m, tuple([target, pred]), 1)

    def clear(self):
        self.m = np.zeros([self.nc, self.nc], dtype=np.float32)

    @property
    def precision(self):
        s = self.m.sum(0)
        s[s == 0] = np.inf
        # return np.diag(self.m) / s # by class
        return np.diag(self.m).sum() / s.sum() # all

    @property
    def recall(self):
        s = self.m.sum(1)
        s[s == 0] = np.inf
        return np.diag(self.m) / s

    @property
    def uar(self):
        """Unweighted accuracy.

        Average of recall of every class, as defined in
        Han, Yu, and Tashev, "Speech Emotion Recognition Using Deep Neural
        Network and Extreme Learning Machine."

        """
        return self.recall.mean()

    @property
    def war(self):
        """Weighted accuracy.

        Accuracy of the entire dataset, as defined in
        Han, Yu, and Tashev, "Speech Emotion Recognition Using Deep Neural
        Network and Extreme Learning Machine."

        """
        return np.diag(self.m).sum() / self.m.sum()

    @property
    def F1(self):
        p = self.precision
        r = self.recall
        d = p + r
        d[d == 0] = np.inf
        return 2 * p * r / d

    @property
    def macroF1(self):
        return self.F1.mean()

    @property
    def microF1(self):
        p = self.precision.mean()
        r = self.recall.mean()
        if p + r == 0:
            return 0.
        return 2 * p * r / (p + r)