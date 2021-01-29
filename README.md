# MachineLearning-tools

Implementation of different tools for machine learning.

This project is for a pedagogic purpose and was developp through my different project in machine learning course.

## Install

All the requirement are in the file requirement.txt

```
pip install -r requirement.txt
```

## Content

### AAlorithm

All the algorithm implementation follow the AAlgorithm model.

This model has two main method:
- fit: it prepare your classifier with a training dataset
- predict: give predictions for the given values

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

Implementation of the bayes algorithm following the Gaussian method.



## Authors

* **Arthur LANG** - *Initial work* - [LangArthur](https://github.com/LangArthur)