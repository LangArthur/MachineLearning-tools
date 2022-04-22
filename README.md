# MachineLearning-tools

Implementation of different tools for machine learning.

This project is for a pedagogic purpose and was developp through my different projects / assignments in machine learning course.
There are also some algorithms that was developped later.

## Install

All the requirement are in the file requirement.txt. I personnaly used pip to install everything.

```
pip install -r requirement.txt
```

## Content

### AAlorithm

All the algorithm implementation follow the AAlgorithm model.

This model has three main methods:
- fit: it prepare your classifier with a training dataset
- predict: give predictions for the given values
- predict_proba: give the probability of the prediction

### Knn Algorithm

Implementation of the K-Neirest Neighbor algorithm.

#### simple example

data variable is the raw dataset
target variable is the classes of your data
guess variable is the element you want to predict a class to
```
data = [[0, 2], [1, 2], [2, 3], [3, 5]]
target = [1, 1, 0, 0]
guess = [[1, 1]]

neigh = MyKNeirestNeighbor(3)
neigh.fit(data, target)
print("predicted class: {}".format(neigh.predict(guess)))
```

### Naive Bayes Algorithm

Implementation of the bayes algorithm following the Gaussian method. I choose this method because it was the simplest one and because it seems the most accurate to treat number values (like in the iris dataset).

Note that this algorithm handle multi-class problem.

### Decision Tree Classifier

Implementation of a tree classifier. It evaluate the impurity by using the Gini ratio.
The decision tree classifier implement the methods from AAlgorithm. Note that it has another function to display the tree.

### Multy-layer Neural Network

/!\ Work in progress
Implmentation of a neural network classifier. It uses backpropagation for adjusting the weights.

### Evaluation

First, here is a structure containing the result of an evaluation

```
class EvaluationResult:
    accuracy: float
    precision: {str, float}
    recall: float
    confusionMatrix: ConfusionMatrix
```

Several methods for evaluations are implemented. The most usefull are the following:

- evaluate: return an object containing different values from the evaluation (see paragraphe on EvaluationResult)
- crossValidation: do a cross-validation on the result with a specific algorithm. Note that the cross-validation handle multi-class classification (if the provide algorithm does).
- rocEvaluation: draw the roc curve and compute the Area Under the Curve (AUC). I tried to implemanted the multi-class gestion for it but I'm not totally sure of the result.

## Authors

* **Arthur LANG** - *Initial work* - [LangArthur](https://github.com/LangArthur)