#!/usr/bin/python3
#
# Created on Wed Jan 27 2021
#
# Arthur Lang
# evaluation.py
#

import numpy
import matplotlib.pyplot

from src.evaluationDataStructure import EvaluationResult, ConfusionMatrix

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
    name = _getEvaluationName(prediction, reality) #TODO rework this part
    res = EvaluationResult()
    res.confusionMatrix.reserve(name)
    for predict, real in zip(prediction, reality):
        res.confusionMatrix.add(predict, real)
    res.accuracy = evaluateAccuracy(res.confusionMatrix, len(reality))
    res.precision = evaluatePrecision(res.confusionMatrix)
    res.recall = evaluateRecall(res.confusionMatrix)
    return res

# return the names of the class of the evaluation
def _getEvaluationName(prediction, reality):
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

## evaluatePrecision
# compute the precision
# @param confMatrix confusion matrix of the experiment
def evaluatePrecision(confMatrix):
    res = {}
    for i, label in enumerate(confMatrix.labels):
        res[label] = confMatrix.data[i][i] / sum(confMatrix.data[i])
    return res

## evaluateRecall
# compute the recall
# @param confMatrix confusion matrix of the experiment
def evaluateRecall(confMatrix):
    res = {}
    for i, label in enumerate(confMatrix.labels):
        res[label] = confMatrix.data[i][i] / sumColumn(confMatrix.data, i)
    return res

## sumColumn
# sum a column in an array
# @param array array with the column
# @param i index of the column to sum
def sumColumn(array, i):
    res = 0
    for elem in array:
        res += elem[i]
    return res

## DrawRoc
# draw the roc curve
# @param probaPrediction probability of the prediction.
# @param reality list of the real class
# @param sizePartition size of the partition for the roc evaluation
# @param classToDisplay class you want to display if the prediction has multiple class. Note that this parameter is not used for the moment
def drawRoc(probaPrediction, reality, sizePartition = 100, classToDisplay=None):
    fprList, tprList = rocEvaluation(probaPrediction, reality, sizePartition)
    _rocGraph(fprList, tprList)

## rocEvaluation
# draw the roc curve
# /!\ for the moment the roc curve consider a two class classification
# @param probaPrediction probability of the prediction.
# @param reality list of the real class
# @param sizePartition size of the partition for the roc evaluation
def rocEvaluation(probaPrediction, reality, sizePartition = 100):
    if (len(numpy.unique(reality)) > 2):
        raise RuntimeError("Error: Roc curve evaluation is not implemented for multiclasse yet.")
    thresholds = [i / sizePartition for i in range(0, sizePartition)]
    tprList = []
    fprList = []

    for threshold in thresholds:
        tpr, fpr = _getTprAndFpr(probaPrediction, reality, threshold)
        tprList.append(tpr)
        fprList.append(fpr)
    return fprList, tprList

## _getTprAndFpr
# @return the True positive rate and the False positive rate
# @param probaPrediction probability of the prediction.
# @param reality list of the real class
# @param threshold threshold on which the 
def _getTprAndFpr(probaPrediction, reality, threshold):
    tp = tn = fp = fn = 0
    for pred, real in zip(probaPrediction, reality):
        if (pred > threshold):
            if (real == 1):
                tp += 1
            else:
                fp += 1
        else:
            if (real == 0):
                tn += 1
            else:
                fn += 1
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    return tpr, fpr

## _rocGraph
# plot the graph for the roc curve
# @param x the true postive rate points
# @param y the false positive rate points
# @param classToDisplay class you want to display if the prediction has multiple class. Note that this parameter is not used for the moment
def _rocGraph(x, y, classToDisplay=None):
    matplotlib.pyplot.ylim(0, 1)
    matplotlib.pyplot.xlim(0, 1)
    matplotlib.pyplot.scatter(x, y)
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.ylabel('True Positive')
    matplotlib.pyplot.xlabel('False Positive')
    matplotlib.pyplot.show()
