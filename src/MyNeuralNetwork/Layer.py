#
# Created on Fri Feb 26 2021
#
# Arthur Lang
# Layer.py
#

import numpy

from src.MyNeuralNetwork.Node import Node

class Layer():
    def __init__(self, nbNodes, activation):
        self._nbNodes = nbNodes
        self._activationFctName = activation
        self._nodes = numpy.empty(shape=(self._nbNodes,), dtype=Node)
        for i in range(self._nbNodes):
            self._nodes[i] = Node(activation, 1) #TODO set weight

    def __str__(self):
        res = "Layer\nNumber of node: " + str(self._nbNodes) + "\tactivation function: " + self._activationFctName
        return res

    def size(self):
        return len(self._nodes)