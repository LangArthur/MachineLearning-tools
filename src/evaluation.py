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

def splitDatasetForCV(data, target, i):
    tmp = list(zip(data, target))
    testingSet = tmp.pop(i)
    testData, testTarget = testingSet[0], testingSet[1]
    data, target = zip(*tmp)
    return numpy.array(data), numpy.array(target), testData, testTarget
    # trainingset = numpy.concatenate(tmp)

## crossValidation
# apply a cross validation on a model
# @param nbFolds nb of folds in the cross validation
# @dataset a dataset formated as the sklearn data set # TODO change this for more modularty
# @algorithm algorithm that must heritate from AAlgorithm (see AAlgorithm file)
def crossValidation(nbFolds, dataset, algorithm, drawRoc=False):
    # check nbFolds
    if (nbFolds < 2):
        raise ValueError("Error: can't do a cross validation with less than 2 folds")
    # shuffle the datas
    indices = numpy.arange(dataset.data.shape[0])
    numpy.random.shuffle(indices)
    # split in folds
    data = numpy.array_split(dataset.data[indices], nbFolds)
    target = numpy.array_split(dataset.target[indices], nbFolds)

    foldEvaluations = []

    # execute algorithm
    for i in range(nbFolds):
        cvData, cvTarget, cvTestData, cvTestTarget = splitDatasetForCV(data, target, i)
        algorithm.fit(cvData, cvTarget)
        if (drawRoc):
            predict = algorithm.predict_proba(cvTestData)
            foldEvaluations.append(predict)
        else:
            predict = algorithm.predict(cvTestData)
            myeval = evaluate(predict, cvTestTarget)
            foldEvaluations.append(myeval)
    if (drawRoc):
        merge = _mergeCrossValidationProba(foldEvaluations, nbFolds)
        rocEvaluation(merge, cvTestTarget)
    else:
        return _mergeCrossValidationRes(foldEvaluations, nbFolds)

## _mergeCrossValidationRes
# merge all the results from the cross-validation
# @param resArray array with all the EvaluationResult
# @param nb of folds give in the cross validation
def _mergeCrossValidationRes(resArray, nbFolds): #TODO rework the merge method (too much loop, could be more optimal -> check numpy method)
    res = EvaluationResult()
    res.confusionMatrix.reserve(resArray[0].confusionMatrix.labels)
    matrixData = []
    # sum all the values from different evaluations
    for evalRes in resArray:
        res.accuracy += evalRes.accuracy
        for key, value in evalRes.precision.items():
            if (key in res.precision):
                res.precision[key] += value
            else:
                res.precision[key] = value
        for key, value in evalRes.recall.items():
            if (key in res.recall):
                res.recall[key] += value
            else:
                res.recall[key] = value
        if (matrixData == []):
            matrixData = evalRes.confusionMatrix.data
        else:
            for i in range(len(matrixData)):
                for j in range(len(matrixData[0])):
                    matrixData[i][j] += evalRes.confusionMatrix.data[i][j]
    # divide each elem by nb of folds
    for i in range(len(matrixData)):
        for j in range(len(matrixData[0])):
            matrixData[i][j] /= nbFolds - 1
    res.confusionMatrix.data = copy.copy(matrixData)
    res.accuracy /= nbFolds - 1
    for key, _ in res.precision.items():
        res.precision[key] /= nbFolds - 1
    for key, _ in res.recall.items():
        res.recall[key] /= nbFolds - 1
    return res

## _mergeCrossValidationProba
# merge probas from a cross-validation
# @param probaArray array with all the proba from the cross validation
# @param nbFolds number of folds used in the cross-validation
def _mergeCrossValidationProba(probaArray, nbFolds):
    for i in range(len(probaArray)):
        if i != 0:
            for j in range(len(probaArray[i])):
                for k in range(len(probaArray[i][j])):
                    probaArray[0][j][k] += probaArray[i][j][k]
    for i in range(len(probaArray[0])):
        for j in range(len(probaArray[0][i])):
            probaArray[0][i][j] /= nbFolds
    return probaArray[0]

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
# @param classToDisplay class you want to display if the prediction has multiple class. Note that this parameter is not used for the moment
def rocEvaluation(probaPrediction, reality, sizePartition = 100, classToDisplay=None):
    fprList, tprList = _getRocEvaluationCoordinate(probaPrediction, reality, sizePartition)
    auc = _computeAuc(fprList, tprList)
    _rocGraph(fprList, tprList, round(auc, 2))

## _getRocEvaluationCoordinate
# return the coordinates for the Roc curve
# @param probaPrediction probability of the prediction.
# @param reality list of the real class
# @param sizePartition size of the partition for the roc evaluation
def _getRocEvaluationCoordinate(probaPrediction, reality, sizePartition = 100):
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
# @param threshold threshold on which the roc is computed
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

## _computeAuc
# compute the area under the curve
# I approximate the area between two values on the curve as a trapeze
# @param xList all the x of the points
# @param yList all the y of the points
def _computeAuc(xList, yList):
    res = 0
    prev_x = None
    prev_y = None
    for x, y in zip(xList, yList):
        if (prev_x != None):
            res += abs(x - prev_x) * prev_y + ((abs(x - prev_x) * abs(y - prev_y)) / 2)
        prev_x = x
        prev_y = y
    return res

## _rocGraph
# plot the graph for the roc curve
# @param x the true postive rate points
# @param y the false positive rate points
# @param classToDisplay class you want to display if the prediction has multiple class. Note that this parameter is not used for the moment
def _rocGraph(x, y, auc, classToDisplay=None):
    # add the origin point to display the curve
    x.append(0)
    y.append(0)
    matplotlib.pyplot.ylim(-0.05, 1.05)
    matplotlib.pyplot.xlim(-0.05, 1.05)
    matplotlib.pyplot.scatter(x, y)
    matplotlib.pyplot.plot(x, y)
    matplotlib.pyplot.plot([0, 1], [0, 1], color='red')
    matplotlib.pyplot.ylabel('True Positive')
    matplotlib.pyplot.xlabel('False Positive')
    matplotlib.pyplot.text(0.9, 0.9, "AUC: " + str(auc), horizontalalignment='center', verticalalignment='center')
    matplotlib.pyplot.show()
