#
# Created on Mon Feb 22 2021
#
# Arthur Lang
# MyNeuralNetwork.py
#

import sys

from src.AAlgorithm import AAlgorithm

class Node():

    def __init__(self, weights):
        self.weights = weights
        self.output = 0
        self.delta = 0 # reprensent backward propagation error

    def activate(self):
        pass

# class Layer():
#      def __init__(self, nbNode)

class MyNeuralNetwork(AAlgorithm):

    def __init__(self):
        super().__init__("Neural-Network")
        self._layerNb = 1 # number of hidden layers in the network
        self._nbNeurone = 10 # number of neurones in each layers
        self._weights = []

    def setHiddenLayerNb(self, nb):
        if (nb > 0):
            self._layerNb = nb
        else:
            print("Error: invalide number of layers: " + str(nb) + " is not > 0.", file=sys.stderr)

    def setNbNeurones(self, nb):
        if (nb > 1): # TODO check if can work with 1
            self._nbNeurone = nb
        else:
            print("Error: invalide number of neurones: " + str(nb) + ".", file=sys.stderr)

    def setWeights(self, weights):
        self._weights = weights

    def fit(self, data, targets):
        pass

    def predict(self, toTest):
        pass

    def predict_proba(self, toTest):
        pass