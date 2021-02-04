#!/usr/bin/python3
#
# Created on Fri Jan 29 2021
#
# Arthur Lang
# MyNaiveBayes.py
#

import numpy
from enum import Enum

from src.AAlgorithm import AAlgorithm

## NaiveBayesType
# enum for type of naive algorithm implemented
class NaiveBayseType(Enum):
    GAUSSIAN = 1 # Implementation of the Gaussian algorithm because it's look the more global one.

class MyNaiveBayes(AAlgorithm):

    def __init__(self, algoType):
        super().__init__("Naive-Bayes")
        self.type = algoType
        self.algoChoice = {
            NaiveBayseType.GAUSSIAN: self._gaussianfit
        }
        self._classes = []
        self._prior = []
        self._mean = []
        self._var = []

    def fit(self, data, target):
        self.algoChoice[self.type](data, target)
        self._isFit = True

    def _setConditionnalProb(self, data, target):
        pass

    def _gaussianfit(self, data, target):
        # find occurence of each class
        self._classes = numpy.unique(target)
        # for each class, fill its mean, var and prior probability
        for dataClass in self._classes:
            selectedData = data[target == dataClass]
            self._mean.append(selectedData.mean(axis=0))
            self._var.append(selectedData.var(axis=0))
            self._prior.append(selectedData.shape[0] / len(target))

    def predict(self, testSample):
        res = []
        for x in testSample:
            posteriors = []
            for i in range(0, self._classes.shape[0]):
                classCondProb = numpy.prod(self._pdf(x, self._mean[i], self._var[i]))
                posteriors.append(classCondProb * self._prior[i])
            res.append(self._classes[numpy.argmax(posteriors)])
        return res

    def predict_proba(self, testSample):
        res = []
        for x in testSample:
            posteriors = []
            for i in range(0, self._classes.shape[0]):
                classCondProb = numpy.prod(self._pdf(x, self._mean[i], self._var[i]))
                posteriors.append(classCondProb * self._prior[i])
            # normalize class probability
            res.append([p / sum(posteriors) for p in posteriors])
        return res

    def _pdf(self, x, mean, var):
        num = numpy.exp(-((x - mean) ** 2 / (2 * var) ** 2))
        den = numpy.sqrt(2 * numpy.pi * var)
        return num / den
