
#!/usr/bin/env python3

from sklearn import datasets

from src.evaluation import *
from src.MyNeuralNetwork.MyNeuralNetwork import *

def main():

    dataset = datasets.load_iris()
    trainingData, testData, trainingLabel, testLabel = partitionningDataset(dataset.data, dataset.target, 80)

    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    # clf.fit(trainingData, trainingLabel)
    # pred = clf.predict(testData)
    # print(evaluate(pred, testLabel))
    nn = MyNeuralNetwork()
    nn.addLayer(6, 'sigmoid')
    nn.addLayer(10, 'sigmoid')
    nn.addLayer(10, 'sigmoid')
    nn.fit(trainingData, trainingLabel)
    return 0

if __name__ == "__main__":
    main()