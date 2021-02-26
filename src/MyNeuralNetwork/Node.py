#
# Created on Fri Feb 26 2021
#
# Arthur Lang
# Node.py
#

import math
import numpy

class Node():

    def __init__(self, activationFctName, weight):
        self._weight = weight
        self.output = 0
        self.delta = 0 # reprensent backward propagation error
        self._bias = numpy.random.rand(1) #TODO maybe change this to just a random number
        self._activationFctList = {
            'sigmoid': self._sigmoid
        }
        if (activationFctName in self._activationFctList):
            self.activate = self._activationFctList[activationFctName]
        else:
            raise RuntimeError("Error: the activation function \'" + activationFctName + "\' is not implemented or do not exist.")

    def _sigmoid(self, x):
        return 1 / 1 + math.exp(x)

    def _sigmoidDerivative(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    # TODO find a way to abstract activation function
    # def activate(self, x):
    #     raise RuntimeError("Error: no activation specified.")

    def run(self, input):
        inputValue = [layer.output for layer in input]
        dot = numpy.sum(inputValue * self._weight) + self._bias
        self.output = self._sigmoid(dot)
