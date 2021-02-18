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

## DescisionTreeNodeType
# enum for all type of leaf in the decision tree
class DescisionTreeNodeType(Enum):
    BOOLEAN = 1
    NUMBER = 2
    RANK = 3

## DescisionTreeNode
# node for the dcision Tree
# note that for the moment, only the number are handle and by default, first split is under the value
class DescisionTreeNode:
    def __init__(self, nodeType, attribute, idx, value):
        self.type = nodeType # type of the node
        self.attribute = attribute # attribute on wich the data is split
        self.attributeIdx = idx # index of the attribute in the data
        self.value = value # value of the split
        self.children = [] # childrens of the node

    def __str__(self):
        res = "type of Node: " + str(self.type)
        res += "\nsplit attribute " + str(self.attribute) + " at the value " + str(self.value)
        res += "\nIt has " + str(len(self.children)) + " children:"
        for node in self.children:
            res += node.__str__()
        return res

## BestSplit
# data structure containing all informations needed for evaluating the best node
@dataclass
class BestSplit:
    giniRatio: int # value of the gini ratio on the concerned class
    attributeIdx: int # index of the attribute (in the targets array). Set to -1 by default
    attributeType: DescisionTreeNodeType # type of the split
    splitValue: int # value of the split

    # def __init__(self, gini, attributeIdx, attributeType, splitValue):
    #     self.giniRatio = gini
    #     self.attributeIdx = attributeIdx
    #     self.attributeType = attributeType
    #     self.splitValue = splitValue

    ## update
    # update the content of a BestSplit dataclass
    # @param giniRation new gini ration
    # @param attribute new attribute
    # @param attributeType new type
    # @prama splitValue new split value
    def update(self, giniRatio, attribute, attributeType, splitValue):
        self.giniRatio = giniRatio
        self.attribute = attribute
        self.attributeType = attributeType
        self.splitValue = splitValue

## DecisionTree
# implementation of the Decision tree
# The tree is binary for the moment
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

    ## _giniRation
    # compute the gini ration
    # @param targets the targets from wich you want the gini ratio
    def _giniRatio(self, targets):
        sum = 0
        labels = numpy.unique(targets)
        for label in labels:
            sum += (numpy.sum(targets == label) / len(targets)) ** 2
        return 1 - sum

    ## fit
    # call to train the tree
    # @param data data to use in train
    # @param targets targets to use in train
    def fit(self, data, targets):
        self._buildTree(data, targets)

    ## _buildTree
    # function that build the tree. This function is recursivly call
    # @param data data that will be used for creating the node
    # @param targets targets associated with the data
    # @param parent give if you want to attache new created node to an existing one
    def _buildTree(self, data, targets, parent=None):
        print(len(data))
        input()
        best = self._getBestAttributeScore(data, targets)
        print(best)
        # if we found a best, create the new tree node
        if (best != None):
            newNode = self._createNode(best, parent)
            if (best.giniRatio != 1):
                firstPartData, secondPartData = self._splitData(list(zip(data, targets)), best.splitValue, best.attributeIdx)
                firstData, firstTargets = numpy.array([ a for a, _ in firstPartData ]), numpy.array([ b for _, b in firstPartData ])
                secondData, secondTargets = numpy.array([ a for a, _ in secondPartData ]), numpy.array([ b for _, b in secondPartData ])
                print("First")
                print(firstData)
                print("Second")
                print(secondData)
                # recursive calls for the two new splited data
                self._buildTree(firstData, firstTargets, newNode)
                self._buildTree(secondData, secondTargets, newNode)

    ## _getBestAttributeScore
    # return the evaluation score of the best split (BestSplit dataclass)
    # @param data data to be evaluate
    # @param targets targets associated with the data
    def _getBestAttributeScore(self, data, targets):
        attributes = numpy.unique(targets)
        best = None
        # test all the attributes
        for i in range(len(attributes)):
            if (len(data) > i + 1):
                score = self._score(data[:, i + 1], targets)
                if (score != None and best == None):
                    best = score
                if (score != None and score.giniRatio < best.giniRatio): #TODO maybe change the method of finding the best split
                    best = score
                    best.attributeIdx = i
        return best

    ## _createNode
    # create a new node and add it in the tree
    # @param score score of the best evaluated split
    # @param parent parent the node should be attach to. Note that is no parent is specified and a tree exist, the node will be skiped
    # @return the new created node
    def _createNode(self, score, parent):
        newNode = DescisionTreeNode(score.attributeType, score.attribute, score.attributeIdx, score.splitValue)
        # set the newNode
        if (self.tree == None):
            self.tree = newNode
        elif (parent != None):
            parent.children.append(newNode)
        return newNode

    ## _splitData
    # split the data on the choosen value
    # @param zipDataSet data ziped with their respective targets
    # @param value value on wich the split should be done
    # @param attributIdx index ot the attribut in the dataset
    def _splitData(self, zipDataSet, value, attributIdx):
        fst = []
        sec = []
        for elem in zipDataSet:
            if (elem[0][attributIdx] < value):
                fst.append(elem)
            else:
                sec.append(elem)
        return fst, sec

    ## _score
    # choose wich type of scoring do depending on the type of the attribute
    # @param column column of data we are interested in
    # @param targets targets associated with the colum
    # @return an instance of BestSplit dataclass containing the best split for the column attribute
    def _score(self, column, targets):
        if (numpy.array_equal(column, column.astype(bool))):
            self._boolData(targets)
        elif (numpy.array_equal(column, column.astype(str))):
            self._rankData(targets)
        elif (map(is_float, column)):
            return self._numericBestSplit(column, targets)
        else:
            raise RuntimeError("Error: the type of data is not handle in this algorithm.")

    ## _boolBestSplit
    # return the best split in the case of boolean attribute
    def _boolBestSplit(self, targets):
        raise RuntimeError("Error: Not implemented yet.")

    ## _numericBestSplit
    # return the best split in the case of numeric attribute
    # @param column column containing the interested attribute
    # @param targets targets from the training set
    # @return return a BestSplit strucutre. Not that attributeIdx is not set in this function
    def _numericBestSplit(self, column, targets):
        # sort the data
        tmp = list(zip(column, targets))
        tmp.sort()
        column, targets = zip(*tmp)
        # set to the second element to skip the two firsts values
        prev = column[1]
        res = None
        for i in range(len(column) - 1):
            j = i + 1
            # skip in the case of the same attribute values
            if (prev != column[j]):
                firstSplit = targets[:j]
                secondSplit = targets[j:]
                # Compute the gini ratio
                firstGini = self._giniRatio(firstSplit) * (len(firstSplit) / len(column))
                secGini = self._giniRatio(secondSplit) * (len(secondSplit) / len(column))
                giniRatio = firstGini + secGini
                # choose to keep or not the attribute
                if (res == None):
                    res = BestSplit(firstGini + secGini, -1, DescisionTreeNodeType(2), (column[i] + column[j]) / 2)
                elif (res.giniRatio > giniRatio):
                    res.update(firstGini + secGini, -1, DescisionTreeNodeType(2), (column[i] + column[j]) / 2)
                prev = column[j]
        return res

    ## _rankBestSplit
    # return the best split in the case of ranking attribute (like a string for example)
    def _rankBestSplit(self, targets):
        raise RuntimeError("Error: Not implemented yet.")

    def predict(self, testSample):
        raise RuntimeError("Error: Not implemented yet.")

    def predict_proba(self, testSample):
        raise RuntimeError("Error: Not implemented yet.")

    def plotTree(self, toPlot):
        raise RuntimeError("Error: Not implemented yet.")
