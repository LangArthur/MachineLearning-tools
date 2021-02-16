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
        prev = column[1]
        for i in range(len(column) - 1):
            j = i + 1
            if (prev != column[j]):
                threshold = (column[i] + column[j]) / 2
                firstSplit = targets[:j]
                print(self._giniRatio(firstSplit))
                secondSplit = targets[j:]
                print(self._giniRatio(secondSplit))
                prev = column[j]

    def _rankData(self, targets):
        print("rank")

    def _giniRatio(self, targets):
        sum = 0
        labels = numpy.unique(targets)
        for label in labels:
            sum += (numpy.sum(targets == label) / len(targets)) ** 2
        return 1 - sum

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