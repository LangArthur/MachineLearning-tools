#!/usr/bin/env python3

import sys

from sklearn import datasets
from src.evaluation import *
from src.DecisionTree import *

def printHelp():
    print("USAGE:\t./main.py dataset")
    print("build a decision tree with a dataset.\n")
    print("DATASET:")
    print(" --iris:\tiris dataset")
    print(" --wine:\twine dataset")
    print(" --cancer:\tbreast cancer dataset")
    print(" --nominal:\tsimple example with nominal attribute")

def main():
    av = sys.argv
    if (len(av) != 2):
        printHelp()
        exit(1)
    if (av[1] != "--nominal"):
        datasetLoader = {
            "--iris": datasets.load_iris,
            "--wine": datasets.load_wine,
            "--cancer": datasets.load_breast_cancer,
        }
        dataset = datasetLoader[av[1]]()

        trainingData, testData, trainingLabel, testLabel = partitionningDataset(dataset.data, dataset.target, 80)
    else:

        trainingData = numpy.array([["green", 30], ["green", 29], ["red", 10], ["red", 28], ["red", 30]])
        trainingLabel = numpy.array([0, 0, 1, 1, 0])
        testData = numpy.array([["green", 27], ["red", 29]])
        testLabel = numpy.array([0, 0])

    dt = DecisionTree()
    dt.fit(trainingData, trainingLabel)
    dt.plotTree()
    pred = dt.predict(testData)
    print(evaluate(pred, testLabel))
    return 0

if __name__ == "__main__":
    main()