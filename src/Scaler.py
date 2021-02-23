#
# Created on Tue Feb 23 2021
#
# Arthur Lang
# Scaler.py
#

import numpy
import copy

class Scaler():

    def __init__(self):
        self._minVals = []
        self._maxVals = []

    def fit(self, data):
        if (len(data) < 1):
            raise RuntimeError("Error: no data pass to fit scaler.")
        self._minVals = copy.copy(data[0])
        self._maxVals = copy.copy(data[0])
        for elem in data:
            for i, value in enumerate(elem):
                if (self._minVals[i] > value):
                    self._minVals[i] = value
                if (self._maxVals[i] < value):
                    self._maxVals[i] = value

    def _normalize(self, value, pos):
        return value - self._minVals[pos] / self._maxVals[pos] - self._minVals[pos]

    def transform(self, data):
        numpy.vectorize(data)
        # https://www.oreilly.com/library/view/python-for-data/9781449323592/ch04.html
        return (numpy.array([i for i, x in enumerate(data)]))