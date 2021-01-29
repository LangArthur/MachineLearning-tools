#!/usr/bin/python3
#
# Created on Wed Jan 27 2021
#
# Arthur Lang
# evaluation.py
#

import numpy

from src.evaluationDataStructure import EvaluationResult, ConfusionMatrix

# https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
# https://towardsdatascience.com/classification-metrics-confusion-matrix-explained-7c7abe4e9543
# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

## crossValidation
# apply a cross validation on a model
# @param nbFolds nb of folds in the cross validation
# @dataset a dataset formated as the sklearn data set # TODO change this for more modularty
# @algorithm algorithm that must heritate from AAlgorithm (see AAlgorithm file)
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
    res = EvaluationResult()
    res.confusionMatrix.reserve(foldEvaluations[0].confusionMatrix.labels)
    for evaluation in foldEvaluations:
        res = res + evaluation
    return res

## Partitionning a dataset
# @param data data to be split
# @param target class of the associated data
# @return return the training data (first part), the test data (second part), the training labels and the test label
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

## evaluate
# evalutate the predictions
# @param prediction data get from the prediction
# @param reality real data values
def evaluate(prediction, reality):
    name = _getEvaluationName(reality, prediction) #TODO rework this part
    res = EvaluationResult()
    res.confusionMatrix.reserve(name)
    for predict, real in zip(prediction, reality):
        res.confusionMatrix.add(predict, real)
    res.accuracy = evaluateAccuracy(res.confusionMatrix, len(reality))
    return res

# return the names of the class of the evaluation
def _getEvaluationName(reality, prediction):
    res = []
    for elem in reality:
        if (not(elem in res)):
            res.append(elem)
    for elem in prediction:
        if (not(elem in res)):
            res.append(elem)
    res.sort()
    return res

## evaluateAccuracy
# compute the accuracy
# @param confMatrix confusion matrix of the experiment
# @param size size of the total number of experiment
def evaluateAccuracy(confMatrix, size):
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
