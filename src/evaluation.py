#!/usr/bin/python3
#
# Created on Wed Jan 27 2021
#
# Arthur Lang
# evaluation.py
#

import numpy

from src.EvaluationResult import *

# https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
# https://towardsdatascience.com/classification-metrics-confusion-matrix-explained-7c7abe4e9543
# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

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
    data = numpy.array_split(dataset.data[indices], nbFolds)
    target = numpy.array_split(dataset.target[indices], nbFolds)
    algorithm.fit(data[0], target[0])
    foldEvaluations = []
    # execute algorithm
    for i in range(nbFolds - 1):
        predict = algorithm.predict(data[i + 1])
        foldEvaluations.append(algorithm.evaluate(predict, target[i + 1]))
    # merge all the result
    res = EvaluationResult(foldEvaluations[0].confusionMatrix.labels)
    setFirstMatrix = False
    for evaluation in foldEvaluations:
        res.accurancy += evaluation.accurancy
        res.precision += evaluation.precision
        res.recall += evaluation.recall
        if (not(setFirstMatrix)):
            res.confusionMatrix = evaluation.confusionMatrix
            setFirstMatrix = True
        else:
            res.confusionMatrix.mean(evaluation.confusionMatrix)
    res.accurancy /= len(foldEvaluations)
    res.precision /= len(foldEvaluations)
    res.recall /= len(foldEvaluations)
    return res

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

