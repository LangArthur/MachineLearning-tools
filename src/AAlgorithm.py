#!/usr/bin/python3
#
# Created on Fri Jan 29 2021
#
# Arthur Lang
# AAlgorithm.py
#

from abc import ABC, abstractmethod

## AAlgorithm
# abstract class for algorithm implementation
class AAlgorithm(ABC):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self._isFit = False # set to true when a dataset is fit

    @abstractmethod
    def fit(self, data, target):
        pass

    @abstractmethod
    def predict(self, testSample):
        pass
