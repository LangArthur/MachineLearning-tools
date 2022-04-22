#!/usr/bin/env python3

from sklearn import datasets
from src.evaluation import *
from src.MyNaiveBayes import *

## Assignment 1
def main():
    dataset = datasets.load_iris()
    data = dataset.data
    target = dataset.target
    # data set for 2 class only
    trainingData, testData, trainingLabel, testLabel = partitionningDataset(data, target, 80)

    myNB = MyNaiveBayes(NaiveBayseType.GAUSSIAN)

    print("Running the cross validation:\n")
    print(crossValidation(10, dataset.data, dataset.target, myNB))

    print("Running the cross validation with roc curve:\n")
    crossValidation(10, data, target, myNB, True)
    return 0

if __name__ == "__main__":
    main()