#!/usr/bin/python3
#
# Created on Wed Jan 27 2021
#
# Arthur Lang
# EvaluationResult.py
#

import copy

from dataclasses import dataclass

## ConfusionMatrix
# Implement a confusion matrix for storing performance of a model
class ConfusionMatrix():

    ## Constructor
    # @param self object pointer
    # @param labels contain the classes name used in the model.
    # It allow to directly instanciate the matrix with the good size
    def __init__(self):
        self.labels = []
        self.data = []

    def __str__(self):
        res = ""
        for value, label in zip(self.data, self.labels):
            res += str(label) + "\t"
            for v in value:
                res += str(v) + "\t"
            res += '\n'
        return res

    def reserve(self, labels):
        self.labels = labels
        self.data = [[ 0 for _ in labels] for _ in labels]

    ## add
    # ædd elem to the matrix
    # @param self object pointer
    # @param predict predicted value by the model
    # @param reality real value
    # TODO expend the matrix if the class is unknown
    def add(self, predict, reality):
        realPos = self.labels.index(reality)
        predictPos = self.labels.index(predict)
        if (realPos == -1 or predictPos == -1):
            raise ValueError("Error: one of the values is not in the label")
        self.data[realPos][predictPos] += 1

    ## len
    # @param self object pointer
    # @return the size of the matrix
    def len(self):
        return len(self.labels)

    ## mean
    # @param self object pointer
    # @param other other matrix to do the mean
    # @return the mean matrix
    def mean(self, other):
        if (self.data == []):
            self = copy.copy(other)
            return self
        else:
            if (self.len() != other.len()):
                raise ValueError("Error: you can't do the mean on two diffenrent size matrix")
            res = ConfusionMatrix()
            res.reserve(self.labels)
            for i in range(self.len()):
                for j in range(self.len()):
                    res.data[i][j] = self.data[i][j] + other.data[i][j] / 2
            return res

@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    confusionMatrix: ConfusionMatrix

    def __init__(self):
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.confusionMatrix = ConfusionMatrix()

    def __str__(self):
        res = "Result of the evalutation:\n\taccuracy:\t" + str(self.accuracy) + "\n\tprecision\t" + str(self.precision)
        res += "\n\trecall:\t\t" + str(self.recall) + "\n\nConfusion Matrix:\n" + self.confusionMatrix.__str__()
        return res

    def __add__(self, other):
        self.accuracy = (self.accuracy + other.accuracy) / 2
        self.precision = (self.precision + other.precision) / 2
        self.recall = (self.recall + other.recall) / 2
        self.confusionMatrix = self.confusionMatrix.mean(other.confusionMatrix)
        return self