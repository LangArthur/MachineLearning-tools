#!/usr/bin/python3
#
# Created on Wed Jan 27 2021
#
# Arthur Lang
# evaluation.py
#

import copy
import numpy
import matplotlib.pyplot

from src.evaluationDataStructure import EvaluationResult, ConfusionMatrix
from src.Utilities import sumColumn


## _splitDatasetForCV
# split the data for an iteration of the cross validation
# @param data: data from the dataset
# @param target: labels associated with the data
# @param i: number of the current iteration
# @return return a data for training with their associated labels, and data for testing with their associated labels
def _splitDatasetForCV(data, target, i):
    tmp = list(zip(data, target))
    testingSet = tmp.pop(i)
    testData, testTarget = testingSet[0], testingSet[1]
    data, target = zip(*tmp)
    return numpy.concatenate(data), numpy.concatenate(target), testData, testTarget

## crossValidation
# apply a cross validation on a model
# @param nbFolds: nb of folds in the cross validation
# @param data data: from the dataset
# @param target: target associated with the data
# @param algorithm: algorithm that must heritate from AAlgorithm (see AAlgorithm file)
# @param drawRoc: boolean if the crossvalidation should display Roc curve
def crossValidation(nbFolds, data, target, algorithm, drawRoc=False):
    # check nbFolds
    if (nbFolds < 2):
        raise ValueError("Error: can't do a cross validation with less than 2 folds")
    # shuffle the datas
    indices = numpy.arange(data.shape[0])
    numpy.random.shuffle(indices)
    # split in folds
    data = numpy.array_split(data[indices], nbFolds)
    target = numpy.array_split(target[indices], nbFolds)

    foldEvaluations = []
    foldsReality = [] #use only for the roc curve
    # execute algorithm
    for i in range(nbFolds):
        # split test from training
        cvData, cvTarget, cvTestData, cvTestTarget = _splitDatasetForCV(data, target, i)
        algorithm.fit(cvData, cvTarget)
        # predict the proba in case you want a roc curve
        if (drawRoc):
            predict = algorithm.predict_proba(cvTestData)
            foldEvaluations.extend(predict)
            foldsReality.extend(cvTestTarget)
        # predict lables if you want a basic crossvalidation
        else:
            predict = algorithm.predict(cvTestData)
            myeval = evaluate(predict, cvTestTarget)
            foldEvaluations.append(myeval)
    if (drawRoc):
        rocEvaluation(numpy.array(foldEvaluations), foldsReality)
    else:
        return _mergeCrossValidationRes(foldEvaluations, nbFolds)

## _mergeCrossValidationRes
# merge all the results from the cross-validation
# @param resArray array with all the EvaluationResult
# @param nb of folds give in the cross validation
def _mergeCrossValidationRes(resArray, nbFolds):
    res = EvaluationResult()
    res.confusionMatrix.reserve(resArray[0].confusionMatrix.labels)

    res.confusionMatrix.data = numpy.array(res.confusionMatrix.data)
    res.confusionMatrix.labels = copy.copy(resArray[0].confusionMatrix.labels)
    # sum all the values from different evaluations
    for evalRes in resArray:
        res.accuracy += evalRes.accuracy
        res.recall = {k: res.recall.get(k, 0) + evalRes.recall.get(k, 0) for k in set(res.recall) | set(evalRes.recall)}
        res.precision = {k: res.precision.get(k, 0) + evalRes.precision.get(k, 0) for k in set(res.precision) | set(evalRes.precision)}
        res.confusionMatrix.data += numpy.array(evalRes.confusionMatrix.data)
    # divide each elem by nb of folds
    res.accuracy /= nbFolds
    for key, _ in res.recall.items():
        res.recall[key] /= nbFolds
    for key, _ in res.precision.items():
        res.precision[key] /= nbFolds
    return (res)

## Partitionning a dataset
# @param data data to be split
# @param target class of the associated data
# @param percent: percentage of data in the training set
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
    name = _getEvaluationName(prediction, reality) #TODO rework this part -> can be done with numpy ?
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
        den = sum(confMatrix.data[i])
        if (den != 0):
            res[label] = confMatrix.data[i][i] / sum(confMatrix.data[i])
        else:
            res[label] = 0
    return res

## evaluateRecall
# compute the recall
# @param confMatrix confusion matrix of the experiment
def evaluateRecall(confMatrix):
    res = {}
    for i, label in enumerate(confMatrix.labels):
        den = sumColumn(confMatrix.data, i)
        if (den != 0):
            res[label] = confMatrix.data[i][i] / den
        else:
            res[label] = 0
    return res

## rocEvaluation
# draw the roc curve
# /!\ for the moment the roc curve consider a two class classification
# @param probaPrediction probability of the prediction.
# @param reality list of the real class
# @param sizePartition size of the partition for the roc evaluation
# @param classToDisplay class you want to display
def rocEvaluation(probaPrediction, reality, sizePartition = 100, classToDisplay=None):
    if (classToDisplay == None):
        classToDisplay = reality[0]
    fprList, tprList = _getRocEvaluationCoordinate(probaPrediction, reality, sizePartition, classToDisplay)
    if (not(0 in fprList and 0 in tprList)):
        fprList.append(0)
        tprList.append(0)
    auc = abs(numpy.trapz(tprList, x=fprList))
    _rocGraph(fprList, tprList, round(auc, 2))

## _getRocEvaluationCoordinate
# return the coordinates for the Roc curve
# @param probaPrediction probability of the prediction.
# @param reality list of the real class
# @param sizePartition size of the partition for the roc evaluation
# @param classToDisplay class you want to display
def _getRocEvaluationCoordinate(probaPrediction, reality, sizePartition, classToDisplay):
    thresholds = [i / sizePartition for i in range(-1, sizePartition + 1)]
    tprList = []
    fprList = []
    labelList = numpy.unique(reality)
    refLabelPos = numpy.where(labelList == classToDisplay)[0]

    for threshold in thresholds:
        tpr, fpr = _getTprAndFpr(reality, numpy.greater_equal(probaPrediction[:,refLabelPos], threshold), classToDisplay)
        tprList.append(tpr)
        fprList.append(fpr)
    return fprList, tprList

## _getTprAndFpr
# @return the True positive rate and the False positive rate
# @param reality: list of the real class
# @param boolThresholds: array of boolean corresponding to the probability greater than the threshold or not
# @param refTarget: class on wich the tpr and fpr are based on
def _getTprAndFpr(reality, boolThresholds, refTarget):
    tp = tn = fp = fn = 0
    for i, real in enumerate(reality):
        if (boolThresholds[i]):
            if (real == refTarget):
                tp += 1
            else:
                fp += 1
        else:
            if (real == refTarget):
                fn += 1
            else:
                tn += 1
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    return tpr, fpr

## _rocGraph
# plot the graph for the roc curve
# @param x the true postive rate points
# @param y the false positive rate points
# @param classToDisplay class you want to display if the prediction has multiple class. Note that this parameter is not used for the moment
def _rocGraph(x, y, auc, classToDisplay=None):
    matplotlib.pyplot.ylim(-0.05, 1.05)
    matplotlib.pyplot.xlim(-0.05, 1.05)
    matplotlib.pyplot.scatter(x, y)
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.plot([0, 1], [0, 1], color='red')
    matplotlib.pyplot.ylabel('True Positive')
    matplotlib.pyplot.xlabel('False Positive')
    matplotlib.pyplot.text(0.9, 0.9, "AUC: " + str(auc), horizontalalignment='center', verticalalignment='center')
    matplotlib.pyplot.show()
