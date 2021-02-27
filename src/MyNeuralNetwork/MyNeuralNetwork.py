#
# Created on Mon Feb 22 2021
#
# Arthur Lang
# MyNeuralNetwork.py
#

# about hyperparamters: https://towardsdatascience.com/guide-to-choosing-hyperparameters-for-your-neural-networks-38244e87dafe

import sys
import numpy
from enum import Enum

from src.AAlgorithm import AAlgorithm
from src.MyNeuralNetwork.Layer import Layer
from src.Scaler import Scaler

class MyNeuralNetwork(AAlgorithm):

    def __init__(self, weightPath="", learningRate = 1):
        super().__init__("Neural-Network")
        self._scaler = Scaler()
        self._learningRate = learningRate
        self._layers = []
        if (weightPath == ""): # TODO check initialization with weights
            self._weights = []
        else:
            self._weights = self._parseWeights(weightPath)

    def __str__(self):
        res = "Neural network\n\nParameters: " + str(self._learningRate) + "\n"
        if (len(self._layers) > 0):
            for layer in self._layers:
                res += layer.__str__()
        else:
            res += "There is no layer in the network"
        return res

    def _parseWeights(self, path):
        raise RuntimeError("Error: Not implemented yet")

    # def setWeights(self, weights):
    #     self._weights = weights

    def addLayer(self, nbOfNode, activation='sigmoid'):
        self._layers.append(Layer(nbOfNode, activation))

    def _cost(self, pred, reality):
        # compute mean square error
        return 1 / len(pred) * (numpy.square(pred - reality)).sum()

    def _nbOfNodePerLayer(self, inputLen):
        res = [inputLen]
        for l in self._layers:
            res.append(l.size())
        return res

    def fit(self, data, targets):
        labels = numpy.unique(targets)
        self.addLayer(len(labels))
        nbOfNodesPerLayer = self._nbOfNodePerLayer(len(data[0]))
        # use He weights initialization formula
        layerWeights = [numpy.random.rand(nbOfNodes) * numpy.sqrt(2 / (nbOfNodes - 1)) for nbOfNodes in nbOfNodesPerLayer]
        pred = []
        for elem in data:
            pred.append(self._feedForward(layerWeights, self._scaler.normalize(elem), labels))
        print("First round prediction:")
        print(pred)

    def _feedForward(self, weights, values, labels):
        for i, layer in enumerate(self._layers):
            values = layer.run(weights[i], values)
        return (labels[numpy.argmax(values)])

    def predict(self, toTest):
        pass

    def predict_proba(self, toTest):
        pass