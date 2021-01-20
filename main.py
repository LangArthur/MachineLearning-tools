#!/usr/bin/python3
#
# Created on Tue Jan 19 2021
#
# Arthur Lang
# main.py
#

import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

from src.MyKNeirestNeighbor import MyKNeirestNeighbor

def main():

    try:
        x = [[0, 2], [1, 2], [2, 3], [3, 5]]
        y = [1, 1, 0, 0]
        toGuess = [[1, 1]]
        myneigh = MyKNeirestNeighbor(3)
        myneigh.fit(x, y)
        print("My kkn: {}".format(myneigh.predict(toGuess)))

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(x, y)
        print("sklearn: {}".format(neigh.predict(toGuess)))

    except Exception as e:
        print(e, file=sys.stderr)
    # dataset = datasets.load_wine()
    # guess = [6.1, 3.3,  5.9, 1.8]

    # print()
    # print(neigh.predict_proba([[0.9]]))


    # neigh.fit(dataset.data, dataset.target)

    return 0

if __name__ == "__main__":
    main()