#!/usr/bin/python3
#
# Created on Tue Jan 19 2021
#
# Arthur Lang
# main.py
#

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

def main():

    iris = datasets.load_iris()
    guess = [6.1, 3.3,  5.9, 1.8]

    # X = [[0], [1], [2], [3]]
    # y = [0, 0, 1, 1]
    # neigh = KNeighborsClassifier(n_neighbors=3)
    # neigh.fit(iris.data, guess)
    # print(neigh.predict([[1.1]]))
    # print(neigh.predict_proba([[0.9]]))

    return 0

if __name__ == "__main__":
    main()