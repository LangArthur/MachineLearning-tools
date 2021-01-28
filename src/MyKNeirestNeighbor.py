#!/usr/bin/python3
#
# Created on Wed Jan 20 2021
#
# Arthur Lang
# MyK-NeirestNeighbor.py
#

import sys

from src.evaluationDataStructure import EvaluationResult
from src.evaluation import evaluateAccuracy, EvaluationResult

## MyKNeirestNeighbor
# implementation of the k-neirest neighbor algorithm.
# It is inspired by the implementation of sklearn.
class MyKNeirestNeighbor():
    
    ## Constructor:
    # @param self object pointer
    # @param n number of neighbor used in the algorithm
    def __init__(self, n = 0):
        self.n = n
        self.trainedData = [] # list of tuple containing the point associate with its class
        self.isFit = False # set to true when a dataset is fit

    ## setN
    # setter for n
    # @param n new n
    def setN(self, n):
        self.n = n

    ## fit
    # train the algorithm
    # @param self object pointer
    # @param data data to train on
    # @param target name of each data features
    def fit(self, data, target):
        if (self.checkFitData(data, target)):
            for i in range(0, len(data)):
                self.trainedData.append((data[i], target[i]))
            self.isFit = True

    ## checkFitData
    # check if the data given to fit is goodly formated
    # @param self object pointer
    # @param data all the data
    # @param target all the classes for the data (previous parameter)
    # @param return True if everything is good
    def checkFitData(self, data, target):
        if (len(data) < 1):
            raise ValueError("Error: no data is given for the training.")
        if (len(data) != len(target)):
            raise ValueError("Error: the target do not correspond to the training data.")
        elemSize = len(data[0])
        for line in data:
            if (len(line) != elemSize):
                raise ValueError("Error: data are not uniform.")
        return True

    ## predict
    # predict the class label for the element
    # @param self object pointer
    # @param testSample element to predict
    # @return prediction, a list with an element for each value to be predicted
    def predict(self, testSample):
        if (self.isFit != True):
            raise ValueError("Error: no dataset has been fit.")
        res = []
        for sample in testSample:
            dist = []
            for elem in self.trainedData:
                dist.append((self.computeDistance(sample, elem[0]), elem[1]))
            sampleRes = 0
            dist.sort()
            for i in range(0, self.n):
                sampleRes += dist[i][1]
            # print(sampleRes / self.n, round(sampleRes / self.n))
            res.append(int(round(sampleRes / self.n, 0)))
        return res

    ## computeDistance
    # compute the absolute difference of both point.
    # note: I did not use the euclidian distance to prevent overflow
    # @param self object pointer
    # @param p1 first point
    # @param p2 second point
    # @return value
    def computeDistance(self, p1, p2):
        res = 0
        # set size to 1 if the points are not arrays
        size = (1 if (type(p1) is int or type(p1) is float) else len(p1))
        if (size != len(p2)):
            raise ValueError("Error: the point has not the same number of arguments as the data set")
        for i in range(0, size):
            res += ((p2[i] - p1[i]) ** 2)
        return res

    def evaluate(self, prediction, reality):
        name = self._getEvaluationName(reality, prediction)
        res = EvaluationResult()
        res.confusionMatrix.reserve(name)
        for predict, real in zip(prediction, reality):
            res.confusionMatrix.add(predict, real)
        res.accuracy = evaluateAccuracy(res.confusionMatrix, len(reality))
        return res

    def _getEvaluationName(self, reality, prediction):
        res = []
        for elem in reality:
            if (not(elem in res)):
                res.append(elem)
        for elem in prediction:
            if (not(elem in res)):
                res.append(elem)
        res.sort()
        return res