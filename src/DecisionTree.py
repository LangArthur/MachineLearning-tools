#
# Created on Thu Feb 11 2021
#
# Arthur Lang
# DecisionTree.py
#

import pydot
import numpy

from graphviz import Source
from enum import Enum
from dataclasses import dataclass

from src.AAlgorithm import AAlgorithm
from src.Utilities import is_float

## DescisionTreeNodeType
# enum for all type of leaf in the decision tree
class DescisionTreeNodeType(Enum):
    FINAL = 0
    BOOLEAN = 1
    NUMBER = 2
    RANK = 3

## DescisionTreeNode
# node for the dcision Tree
# note that for the moment, only the number are handle and by default, first split is under the value
class DescisionTreeNode:
    def __init__(self, nodeType, attribute, idx, value, nodeId):
        self.type = nodeType # type of the node
        self.attribute = attribute # attribute on wich the data is split
        self.attributeIdx = idx # index of the attribute in the data
        self.finalClass = -1 # final class. It used only with FINAL node
        self.value = value # value of the split
        self.id = nodeId # id of the node
        self.children = [] # childrens of the node

    def __str__(self):
        res = "type of Node: " + str(self.type) + "\n"
        if (self.type != DescisionTreeNodeType.FINAL):
            res += "split attribute " + str(self.attribute) + " at the value " + str(self.value)
            res += "\nIt has " + str(len(self.children)) + " children:\n"
            for node in self.children:
                res += "- " + node.__str__()
        return res

## BestSplit
# data structure containing all informations needed for evaluating the best node
@dataclass
class BestSplit:
    giniRatio: int # value of the gini ratio on the concerned class
    attributeIdx: int # index of the attribute (in the targets array). Set to -1 by default
    attributeType: DescisionTreeNodeType # type of the split
    splitValue: int # value of the split
    finalClass: int # use for remember the class of the split when it's pure.

    ## update
    # update the content of a BestSplit dataclass
    # @param giniRation new gini ration
    # @param attribute new attribute
    # @param attributeType new type
    # @prama splitValue new split value
    def update(self, giniRatio, attributeIdx, attributeType, splitValue):
        self.giniRatio = giniRatio
        self.attributeIdx = attributeIdx
        self.attributeType = attributeType
        self.splitValue = splitValue

## DecisionTree
# implementation of the Decision tree
# The tree is binary for the moment
class DecisionTree(AAlgorithm):

    def __init__(self):
        super().__init__("Descicion tree")
        self.tree = None
        self.availableId = 0
        self._predictionFct = {
            DescisionTreeNodeType.NUMBER: self._predictNum,
            DescisionTreeNodeType.FINAL: lambda node, value : node.value
        }

    def __str__(self):
        res = "Decsision Tree:\n\n"
        if (self.tree == None):
            res += "No data in the tree. Did you forget to fit it to your data?"
        else:
            res += self.tree.__str__()
        return res

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
        # if (parent != None):
        #     print("parent id: " + str(parent.id))
        #     print("data size: " + str(len(data)))
        best = self._getBestAttributeScore(data, targets)
        # if we found a best, create the new tree node
        if (best != None):
            # add the node to the tree
            newNode = self._createNode(best, data[0][best.attributeIdx], parent)
            if (newNode.type != DescisionTreeNodeType.FINAL):
                firstPartData, secondPartData = self._splitData(list(zip(data, targets)), best.splitValue, best.attributeIdx)
                firstData, firstTargets = numpy.array([ a for a, _ in firstPartData ]), numpy.array([ b for _, b in firstPartData ])
                secondData, secondTargets = numpy.array([ a for a, _ in secondPartData ]), numpy.array([ b for _, b in secondPartData ])
                # recursive calls for the two new splited data
                self._buildTree(firstData, firstTargets, newNode)
                self._buildTree(secondData, secondTargets, newNode)

    ## _getBestAttributeScore
    # return the evaluation score of the best split (BestSplit dataclass)
    # @param data data to be evaluate
    # @param targets targets associated with the data
    def _getBestAttributeScore(self, data, targets):
        nbAttributes = len(data[0])
        best = None
        # test all the attributes
        for i in range(nbAttributes):
            if (len(data) > i):
                score = self._score(data[:, i], targets)
                if (score != None):
                    score.attributeIdx = i
                    if (best == None):
                        best = score
                    elif (best.giniRatio > score.giniRatio): #TODO maybe change the method of finding the best split with same gini ratio
                        best = score
        return best

    ## _createNode
    # create a new node and add it in the tree
    # @param score score of the best evaluated split
    # @param attribut of the new node
    # @param parent parent the node should be attach to. Note that is no parent is specified and a tree exist, the node will be skiped
    # @return the new created node
    def _createNode(self, score, attribute, parent):
        newNode = DescisionTreeNode(score.attributeType, attribute, score.attributeIdx, score.splitValue, self.availableId)
        if (newNode.type == DescisionTreeNodeType.FINAL):
            newNode.value = score.finalClass
        self.availableId += 1
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
        # if (numpy.array_equal(column, column.astype(bool))): #TODO fix this condition
        #     self._boolBestSplit(targets)
        if (numpy.array_equal(column, column.astype(str))):
            self._rankBestSplit(targets)
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
        # if the content is pure or you have only one element in the column, return a final node
        if (len(column) == 1 or len(numpy.unique(targets)) == 1):
            return BestSplit(0, -1, DescisionTreeNodeType.FINAL, -1, targets[0])
        # sort the data
        tmp = list(zip(column, targets))
        tmp.sort()
        column, targets = zip(*tmp)
        prev = column[0]
        res = None
        for i in range(len(column) - 1):
            j = i + 1
            # skip in the case of the same attribute values
            if (prev != column[j] and j != len(column)):
                firstSplit = targets[:j]
                secondSplit = targets[j:]
                # Compute the gini ratio
                firstGini = self._giniRatio(firstSplit) * (len(firstSplit) / len(column))
                secGini = self._giniRatio(secondSplit) * (len(secondSplit) / len(column))
                giniRatio = firstGini + secGini
                # choose to keep or not the attribute
                splitValue = round((column[i] + column[j]) / 2, 5) # TODO risk of infinit loop if round is equal to 0
                if (res == None):
                    res = BestSplit(giniRatio, j, DescisionTreeNodeType(2), splitValue, -1)
                elif (res.giniRatio > giniRatio):
                    res.update(giniRatio, j, DescisionTreeNodeType(2), splitValue)
                prev = column[j]
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

    ## _rankBestSplit
    # return the best split in the case of ranking attribute (like a string for example)
    def _rankBestSplit(self, targets):
        raise RuntimeError("Error: Not implemented yet.")

    ## predict
    # predict the class label for the elements
    # @param self object pointer
    # @param testSample elements to predict in an array
    # @return prediction, a list with an element for each value to be predicted
    def predict(self, testSample):
        res = []
        for toTest in testSample:
            res.append(self._moveDownTree(toTest))
        return res

    ## _moveDownTree
    # go through the tree
    # @param value element used through the tree
    # @param node actual node
    # @return the class of value
    def _moveDownTree(self, value, node = None):
        if (node == None):
            node = self.tree
        if (node.type in self._predictionFct):
            return self._predictionFct[node.type](node, value)
        else:
            raise RuntimeError("Error: prediction for " + str(node.type) + " is not implemented yet.")

    ## _predictNum
    # predict the class for a number attribute
    # @param node current node
    # @param value element to compare
    def _predictNum(self, node, value):
        if (len(node.children) < 2):
            raise RuntimeError("Error: An error occured in the tree creation. One of his node do not have enough children.")
        if (value[node.attributeIdx] < node.value):
            return self._moveDownTree(value, node.children[0])
        else:
            return self._moveDownTree(value, node.children[1])

    ## predict_proba
    # get the proba of the prediction (despite the class)
    # @param testSample elements to predict in an array
    # @return a list with all the probabilities predictions in an array
    def predict_proba(self, testSample):
        raise RuntimeError("Error: Not implemented yet.")

    ## plotTree
    # plot the tree
    # @param path path to the output dot file
    # @param legend give a legend to the attributs
    def plotTree(self, path = "output/output.dot", legend = []):
        graph = pydot.Graph("Decision Tree", graph_type='graph', bgcolor='white')
        self._plotNode(graph, legend=legend)
        f = open(path, "w")
        f.write(graph.to_string())
        f.close()
        try:
            s = Source.from_file(path)
            s.view()
        except Exception as e:
            print("Error: Can't display the graph properly. Here is the reason:")
            print(e)
            print("\n\033[91mNote to the teacher:\033[0m")
            print("If it ask you to put the Graphviz executable on your path, I recommande to do sudo 'apt install graphviz on Ubuntu'")
            print("Don't worry, an output file is generated in output directory, so you can visualize it.")

    ## _plotNode
    # add node to the ploted tree
    # @param graph graph of the tree
    # @param node node to plote
    # @param parent parent of the node
    # @param legend give a legod to the attributes
    def _plotNode(self, graph, node = None, parent = None, legend = []):
        if (node == None):
            node = self.tree
        newNode = None
        if (node.type != DescisionTreeNodeType.FINAL):
            if (legend != []):
                newNode = pydot.Node(node.id, shape="circle", label=("class: " + legend[node.attributeIdx] + " split at " + str(node.value)))
            else:
                newNode = pydot.Node(node.id, shape="circle", label=("class: " + str(node.attributeIdx) + " split at " + str(node.value)))
        else:
            newNode = pydot.Node(node.id, shape="circle", label="class: " + str(node.value))
        graph.add_node(newNode)
        if (parent != None):
            graph.add_edge(pydot.Edge(parent.id, node.id))
        for child in node.children:
            self._plotNode(graph, child, node)