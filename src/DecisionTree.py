#
# Created on Thu Feb 11 2021
#
# Arthur Lang
# DecisionTree.py
#

import numpy

from enum import Enum
from dataclasses import dataclass

from src.AAlgorithm import AAlgorithm
from src.Utilities import is_float

class DescisionTreeNodeType(Enum):
    BOOLEAN = 1
    NUMBER = 2
    RANK = 3

class DescisionTreeNode:
    def __init__(self, nodeType, attribute, value):
        self.type = nodeType
        self.attribute = attribute
        self.value = value
        self.children = []

    def __str__(self):
        res = "type of Node: " + str(self.type)
        res += "\nsplit attribute " + str(self.attribute) + " at the value " + str(self.value)
        res += "\nIt has " + str(len(self.children)) + " children:"
        for node in self.children:
            res += node.__str__()
        return res

@dataclass
class BestSplit:
    giniRatio: int
    attribute: int
    attributeType: DescisionTreeNodeType
    splitValue: int

    def update(self, giniRatio, attribute, attributeType, splitValue):
        self.giniRatio = giniRatio
        self.attribute = attribute
        self.attributeType = attributeType
        self.splitValue = splitValue

class DecisionTree(AAlgorithm):

    def __init__(self):
        super().__init__("Descicion tree")
        self.tree = None

    def __str__(self):
        res = "Decsision Tree:\n\n"
        if (self.tree == None):
            res += "No data in the tree. Did you forget to fit it to your data?"
        else:
            res += self.tree.__str__()
        return res

    def _boolData(self, targets):
        print("bool")

    def _numericBestSplit(self, column, targets):
        tmp = list(zip(column, targets))
        tmp.sort()
        column, targets = zip(*tmp)
        prev = column[1]
        ginies = []
        res = None
        for i in range(len(column) - 1):
            j = i + 1
            # skip in the case of the same attribut values
            if (prev != column[j]):
                firstSplit = targets[:j]
                secondSplit = targets[j:]
                firstGini = self._giniRatio(firstSplit) * (len(firstSplit) / len(column))
                secGini = self._giniRatio(secondSplit) * (len(secondSplit) / len(column))
                giniRatio = firstGini + secGini
                if (res == None):
                    res = BestSplit(firstGini + secGini, None, DescisionTreeNodeType(2), (column[i] + column[j]) / 2)
                elif (res.giniRatio > giniRatio):
                    res.update(firstGini + secGini, None, DescisionTreeNodeType(2), (column[i] + column[j]) / 2)
                prev = column[j]
            # if (prev != column[j]):
            #     firstSplit = targets[:j]
            #     secondSplit = targets[j:]
            #     firstGini = self._giniRatio(firstSplit) * (len(firstSplit) / len(column))
            #     secGini = self._giniRatio(secondSplit) * (len(secondSplit) / len(column))
            #     # first member is giniRatio, the second is the target attribute, the third is the value of the split #TODO update this comment
            #     ginies.append([firstGini + secGini, (column[i] + column[j]) / 2, DescisionTreeNodeType(2)])
            #     # ginies.append([firstGini + secGini, (column[i] + column[j]) / 2, firstSplit, secondSplit])
            #     prev = column[j]
        # ginies.sort()
        return res

    def _rankData(self, targets):
        print("rank")

    def _giniRatio(self, targets):
        sum = 0
        labels = numpy.unique(targets)
        for label in labels:
            sum += (numpy.sum(targets == label) / len(targets)) ** 2
        return 1 - sum

    def fit(self, data, targets, names=[]):
        attributes = numpy.unique(targets)
        self._buildTree(data, targets, attributes)

    def _buildTree(self, data, targets, attributes, parent=None):
        best = None
        for i in range(len(attributes)):
            if (len(data) > i):
                score = self._score(data[:, i + 1], targets)
                if (score != None and best == None):
                    best = score
                if (score != None and score.giniRatio < best.giniRatio): #TODO maybe change the method of finding the best split
                    best = score
                    score.attribute = attributes[i]
        if (best != None):
            newNode = DescisionTreeNode(score.attributeType, score.attribute, score.splitValue)
            if (self.tree == None):
                self.tree = newNode
            elif (parent != None):
                parent.children.append(newNode)
            if (score.giniRatio != 1):
                firstPartData, secondPartData = self._splitData(data, newNode)
                self._buildTree(firstPartData, secondPartData, attributes, newNode)

    def _splitData(self, data, node):
        return [], []

    def _score(self, column, targets):
        # print(column)
        if (numpy.array_equal(column, column.astype(bool))):
            self._boolData(targets)
        elif (numpy.array_equal(column, column.astype(str))):
            self._rankData(targets)
        elif (map(is_float, column)):
            return self._numericBestSplit(column, targets)
        else:
            raise RuntimeError("Error: the type of data is not handle in this algorithm.")

    def predict(self, testSample):
        raise RuntimeError("Error: Not implemented yet.")

    def predict_proba(self, testSample):
        raise RuntimeError("Error: Not implemented yet.")

    def plotTree(self, toPlot):
        raise RuntimeError("Error: Not implemented yet.")
