#!/usr/bin/python3
#
# Created on Wed Jan 27 2021
#
# Arthur Lang
# evaluation.py
#

import numpy

from src.EvaluationResult import ConfusionMatrix

## crossValidation
# apply a cross validation on a model
def crossValidation(nbFolds, dataset, algorithm):
    # check nbFolds
    if (nbFolds < 2):
        raise ValueError("Error: can't do a cross validation with less than 2 folds")
    # shuffle the datas
    indices = numpy.arange(dataset.data.shape[0])
    numpy.random.shuffle(indices)
    # split in folds
    data = numpy.split(dataset.data[indices], nbFolds)
    target = numpy.split(dataset.target[indices], nbFolds)
    algorithm.fit(data[0], target[0])
    res = []
    for i in range(nbFolds - 1):
        predict = algorithm.predict(data[i + 1])
        res.append(algorithm.evaluate(predict, target[i + 1]))
    for mat in res:
        print(mat)

def partitionningDataset(data, target, percent):
    if (len(data) < 4):
        raise ValueError("Error: can't split because the dataset is to little.")
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    data = data[indices]
    target = target[indices]
    splitPos = int(len(data) * (percent / 100))
    if (splitPos <= 0):
        splitPos = 1
    return data[:splitPos], data[splitPos:], target[:splitPos], target[splitPos:]

def evaluateAccurancy(confMatrix, size):
    res = 0
    for i in range(0, confMatrix.len()):
        res += confMatrix.data[i][i]
    return res / size

def evaluatePrecision(confMatrix, size):
    pass

def evaluateRecall(confMatrix, size):
    pass

def getMeanSquaredError():
    pass

