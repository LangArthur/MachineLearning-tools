#
# Created on Wed Jan 20 2021
#
# Arthur Lang
# MyK-NeirestNeighbor.py
#

import sys

## MyKNeirestNeighbor
# implementation of the k-neirest neighbor algorithm.
# It is inspired by the implementation of sklearn.
class MyKNeirestNeighbor():
    
    ## Constructor:
    # @param self object pointer
    # @param n number of neighbor used in the algorithm
    def __init__(self, n = 0):
        self.n = n
        self.trainedData = [] #list of tuple containing the point associate with its class
        self.isFit = False

    ## fit
    # train the algorithm
    # @param self object pointer
    # @param n number of neighbor used in the algorithm
    def fit(self, data, target):
        if (self.checkFitData(data, target)):
            for i in range(0, len(data)):
                self.trainedData.append((data[i], target[i]))
            self.isFit = True

    ## checkFitData
    # check if the data given to fit is goodly formated
    # @param self object pointer
    # @param data all the data
    # @param target all the classes for the data (previous parameter)
    def checkFitData(self, data, target):
        if (len(data) < 1):
            raise ValueError("Error: no data is given for the training.")
        if (len(data) != len(target)):
            raise ValueError("Error: the target do not correspond to the training data.")
        elemSize = len(data[0])
        for line in data:
            if (len(line) != elemSize):
                raise ValueError("Error: data are not uniform.")
        return True

    ## predict
    # predict the class label for the element
    # @param self object pointer
    # @param testSample element to predict
    def predict(self, testSample):
        res = []
        for sample in testSample:
            dist = []
            for elem in self.trainedData:
                dist.append(self.computeDistance(sample, elem[0]))
            sampleRes = 0
            dist.sort()
            for i in range(0, self.n):
                sampleRes += self.trainedData[i][1]
            res.append(int(round(sampleRes / self.n, 0)))
        return res

    ## computeDistance
    # compute the squared euclidian distance
    def computeDistance(self, p1, p2):
        res = 0
        # set size to 1 if the points are not arrays
        size = (1 if (type(p1) is int or type(p1) is float) else len(p1))
        if (size != len(p2)):
            raise ValueError("Error: the point has not the same number of arguments as the data set")
        for i in range(0, size):
            res += (p2[i] - p1[i]) ** 2
        return res