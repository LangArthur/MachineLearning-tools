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

def compare(res1, res2):
    print("Compare: ")
    for mk, sk in zip(res1, res2):
        if (mk != sk):
            print("\033[91m{} {}".format(mk, sk) + "\033[0m")
        else:
            print(mk, sk)

def main():
    dataset = datasets.load_iris()

    # dataset = datasets.load_wine()
    # messy case
    # guess = [[1.423e+02, 1.010e+00, 2.430e+20, 1.060e+01, 3.270e+02, 2.800e+00, 3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00, 1.065e+03]]
    # guess = [[1.423e+02, 1.010e+00, 2.430e+02, 1.060e+01, 3.270e+02, 2.800e+00, 3.060e+00, 2.800e-01, 2.290e+00, 5.640e+02, 1.040e+03, 3.920e+00, 1.065e+03]]

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
    # predict = myNB.predict(testData)
    predict = myNB.predict_proba(testData)

    drawRoc(numpy.array(predict)[:,1], testLabel, 10)

    # print(crossValidation(5, dataset, myneigh))

    # myneigh.fit(trainingData, trainingLabel)
    # predict = myneigh.predict(testData)
    # print(myneigh.evaluate(predict, testLabel))

    # compare(testLabel, predict)
    return 0

if __name__ == "__main__":
    main()