#!/usr/bin/env python3
#
# Created on Tue Jan 19 2021
#
# Arthur Lang
# main.py
#

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import sys

from src.MyNaiveBayes import *
from src.evaluation import *
from src.DecisionTree import *
from src.MyNeuralNetwork.MyNeuralNetwork import *


from src.Scaler import Scaler # TODO remove that

## compare
# function for debuging purpose. It show two array and underline the different values in red
# @param res1 first array
# @param res2 second array
def compare(res1, res2):
    print("Compare: ")
    for mk, sk in zip(res1, res2):
        if (mk != sk):
            print("\033[91m{} {}".format(mk, sk) + "\033[0m")
        else:
            print(mk, sk)

## Assignment 1
def main():
    dataset = datasets.load_iris()
    data = dataset.data[:100]
    target = dataset.target[:100]
    # data set for 2 class only
    trainingData, testData, trainingLabel, testLabel = partitionningDataset(data, target, 80)

    myNB = MyNaiveBayes(NaiveBayseType.GAUSSIAN)

    print("Running the cross validation:\n")
    print(crossValidation(10, dataset.data, dataset.target, myNB))

    print("Running the cross validation with roc curve:\n")
    crossValidation(10, data, target, myNB, True)
    return 0

def printHelp():
    print("USAGE:\t./main.py dataset")
    print("build a decision tree with a dataset.\n")
    print("DATASET:")
    print(" --iris:\tiris dataset")
    print(" --wine:\twine dataset")
    print(" --cancer:\tbreast cancer dataset")
    print(" --nominal:\tsimple example with nominal attribute")

## Assignment 2
# def main():
#     av = sys.argv
#     if (len(av) != 2):
#         printHelp()
#         exit(1)
#     if (av[1] != "--nominal"):
#         datasetLoader = {
#             "--iris": datasets.load_iris,
#             "--wine": datasets.load_wine,
#             "--cancer": datasets.load_breast_cancer,
#         }
#         dataset = datasetLoader[av[1]]()

#         trainingData, testData, trainingLabel, testLabel = partitionningDataset(dataset.data, dataset.target, 80)
#     else:

#         trainingData = numpy.array([["green", 30], ["green", 29], ["red", 10], ["red", 28], ["red", 30]])
#         trainingLabel = numpy.array([0, 0, 1, 1, 0])
#         testData = numpy.array([["green", 27], ["red", 29]])
#         testLabel = numpy.array([0, 0])

#     dt = DecisionTree()
#     dt.fit(trainingData, trainingLabel)
#     dt.plotTree()
#     pred = dt.predict(testData)
#     print(evaluate(pred, testLabel))
#     return 0

## Assignment 3
# def main():

#     dataset = datasets.load_iris()
#     trainingData, testData, trainingLabel, testLabel = partitionningDataset(dataset.data, dataset.target, 80)

#     # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
#     # clf.fit(trainingData, trainingLabel)
#     # pred = clf.predict(testData)
#     # print(evaluate(pred, testLabel))
#     nn = MyNeuralNetwork()
#     nn.addLayer(6, 'sigmoid')
#     nn.addLayer(10, 'sigmoid')
#     nn.addLayer(10, 'sigmoid')
#     nn.fit(trainingData, trainingLabel)
#     return 0

if __name__ == "__main__":
    main()