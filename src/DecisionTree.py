#
# Created on Thu Feb 11 2021
#
# Arthur Lang
# DecisionTree.py
#

import numpy

from src.AAlgorithm import AAlgorithm
from src.Utilities import is_float

class DecisionTree(AAlgorithm):

    def __init__(self):
        super().__init__("Descicion tree")
        self._dataTypeFit = {
            type(1): self._numericData,
            type("string"): self._rankData,
            type(bool): self._boolData,
        }

    def _boolData(self, targets):
        print("bool")

    def _numericData(self, column, targets):
        tmp = list(zip(column, targets))
        tmp.sort()
        column, targets = zip(*tmp)
        for i in range(len(column) - 1):
            j = i + 1
            threshold = (column[i] + column[j]) / 2
            # split array following the threshold

    def _rankData(self, targets):
        print("rank")

    def _giniRatio(self, targets):
        sum = 0
        labels = numpy.unique(targets)
        for label in labels:
            sum += (numpy.sum(targets == label) / len(targets)) ** 2
        return 1 - sum

    # def _giniRatio(self, threshold, data, target):
    #     labels = numpy.unique(target)
    #     score = [0 for _ in range(len(labels) * 2)] # score for each feature.
    #     for i, val in enumerate(data):
    #         if (val < threshold):
    #             score[labels.index(target[i])] += 1
    #         else:
    #             score[labels.index(target[i]) + len(labels)] += 1
    #     print(score)
    #     return 0

    def fit(self, data, targets):
        # get best feature
        # for i in range(len(data[0]) - 1):
        i = 0
        self._chooseScoringMethod(data[:,i + 1], targets)

    def _chooseScoringMethod(self, column, targets):
        # print(column)
        if (numpy.array_equal(column, column.astype(bool))):
            self._boolData(targets)
        elif (numpy.array_equal(column, column.astype(str))):
            self._rankData(targets)
        elif (map(is_float, column)):
            self._numericData(column, targets)
        else:
            raise RuntimeError("Error: ")

    def predict(self, testSample):
        pass

    def predict_proba(self, testSample):
        pass

    def plotTree(self, toPlot):
        pass