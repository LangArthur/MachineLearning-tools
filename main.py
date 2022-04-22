#!/usr/bin/env python3
#
# Created on Tue Jan 19 2021
#
# Arthur Lang
# main.py
# A main file for only purpose of testing while developping this repo
#

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


from src.evaluation import *
from src.MyNeuralNetwork.MyNeuralNetwork import *


from src.Scaler import Scaler # TODO remove that

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
