#
# Created on Mon Feb 22 2021
#
# Arthur Lang
# MyNeuralNetwork.py
#

import sys
import numpy
import math
from enum import Enum

from src.AAlgorithm import AAlgorithm

class Node():

    def __init__(self, weights, inputLayer, activationFct):
        self._weights = weights
        self.output = 0
        self.delta = 0 # reprensent backward propagation error
        self._input = inputLayer
        self._bias = numpy.random.rand(1) #TODO maybe change this to just a random number
        self.activate = activationFct

    def _sigmoid(self, x):
        return 1 / 1 + math.exp(x)

    # TODO find a way to abstract activation function
    # def activate(self, x):
    #     raise RuntimeError("Error: no activation specified.")

    def run(self):
        inputValue = [layer.output for layer in self._input]
        dot = numpy.sum(inputValue * self._weights) + self._bias
        self.output = self._sigmoid(dot)

class Layer():
    def __init__(self, nbNode, activation):
        self._nodes = numpy.array([], dtype=Node)

    # pass this function as parameter
    def _sigmoid(self, x):
        return 1 / 1 + math.exp(x)

class MyNeuralNetwork(AAlgorithm):

    def __init__(self, nbLayers = 1):
        super().__init__("Neural-Network")
        self._layerNb = nbLayers # number of hidden layers in the network
        self._neurones = numpy.empty(self._layerNb, 1)
        self._learningRate = 0 #TODO set to a value
        self._layers = []

    def setHiddenLayerNb(self, nb):
        if (nb > 0):
            self._layerNb = nb
        else:
            print("Error: invalide number of layers: " + str(nb) + " is not > 0.", file=sys.stderr)

    def setWeights(self, weights):
        self._weights = weights

    def add(self, nbOfNode, activation='sigmoid'):
        self._layerNb.append(Layer(nbOfNode, activation))
        pass

    def _cost(self, pred, reality):
        # compute mean square error
        return 1 / len(pred) * (numpy.square(pred - reality)).sum()

    def fit(self, data, targets):
        # TODO build the network
        pass

    def predict(self, toTest):
        pass

    def predict_proba(self, toTest):
        pass