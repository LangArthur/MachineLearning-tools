#!/usr/bin/python3
#
# Created on Tue Jan 19 2021
#
# Arthur Lang
# main.py
#

import sys
import random

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

from src.MyKNeirestNeighbor import MyKNeirestNeighbor
from src.MyNaiveBayes import *
from src.evaluation import *

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

def main():
    dataset = datasets.load_iris()
    data = dataset.data[:100]
    target = dataset.target[:100]
    trainingData, testData, trainingLabel, testLabel = partitionningDataset(data, target, 80)
    # trainingData, testData, trainingLabel, testLabel = partitionningDataset(dataset.data, dataset.target, 80)

    # clf = GaussianNB()
    # clf.fit(trainingData, trainingLabel)
    # prediction = clf.predict(testData)
    # prediction = clf.predict_proba(testData)[:, 1]

    myNB = MyNaiveBayes(NaiveBayseType.GAUSSIAN)

    myNB.fit(trainingData, trainingLabel)
    predict = myNB.predict(testData)
    # print(evaluate(predict, testLabel)) 
    predictProba = myNB.predict_proba(testData)

    rocEvaluation(numpy.array(predictProba)[:,1], testLabel, 10)

    # print(crossValidation(5, dataset, myNB))
    return 0

if __name__ == "__main__":
    main()