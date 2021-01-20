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
        # messy case
        dataset = datasets.load_wine()
        guess = [[1.423e+02, 1.010e+00, 2.430e+20, 1.060e+01, 3.270e+02, 2.800e+00, 3.060e+00, 2.800e-01, 2.290e+00, 5.640e+00, 1.040e+00, 3.920e+00, 1.065e+03]]

        # simple case
        # data = [[0, 2], [1, 2], [2, 3], [3, 5]]
        # target = [1, 1, 0, 0]
        # guess = [[1, 1]]

        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(dataset.data, dataset.target)
        print("sklearn: {}".format(neigh.predict(guess)))

        myneigh = MyKNeirestNeighbor(3)
        myneigh.fit(dataset.data, dataset.target)
        print("My kkn: {}".format(myneigh.predict(guess)))
        return 0
    except Exception as e:
        print(e, file=sys.stderr)
        return 1

if __name__ == "__main__":
    main()