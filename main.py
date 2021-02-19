#!/usr/bin/python3
#
# Created on Tue Jan 19 2021
#
# Arthur Lang
# main.py
#

# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

from src.MyNaiveBayes import *
from src.evaluation import *
from src.DecisionTree import *

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

# def main():
#     dataset = datasets.load_iris()
#     data = dataset.data[:100]
#     target = dataset.target[:100]
#     # data set for 2 class only
#     trainingData, testData, trainingLabel, testLabel = partitionningDataset(data, target, 80)

#     myNB = MyNaiveBayes(NaiveBayseType.GAUSSIAN)
#     myNB.fit(trainingData, trainingLabel)

#     print("Running the cross validation:\n")
#     print(crossValidation(10, dataset, myNB))

#     print("Running the cross validation with roc curve:\n")
#     predictProba = myNB.predict_proba(testData)
#     rocEvaluation(numpy.array(predictProba)[:,1], testLabel, 10, True)

#     return 0

def main():
    dataset = datasets.load_iris()

    trainingData, testData, trainingLabel, testLabel = partitionningDataset(dataset.data, dataset.target, 80)
    dt = DecisionTree()
    dt.fit(trainingData, trainingLabel)
    pred = dt.predict(testData)
    print(pred)
    print(evaluate(pred, testLabel))
    return 0


if __name__ == "__main__":
    main()