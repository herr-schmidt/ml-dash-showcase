from matplotlib import pyplot
import numpy as np

from sklearn import datasets
from svm import SVM

a = np.array([1, 2])
b = np.array([4, 5])

iris = datasets.load_iris(as_frame=True)
# X = iris.data[:, :2]  # we only take the first two features.
# Y = iris.target

print(iris.frame.iloc[:, [0, 1, 2, 3, 4]])

# take only first 2 attributes
class_1 = iris.frame[iris.frame["target"] == 0].iloc[:, [0, 1, 4]]
class_2 = iris.frame[iris.frame["target"] == 1].iloc[:, [0, 1, 4]]
class_3 = iris.frame[iris.frame["target"] == 2].iloc[:, [0, 1, 4]]


pyplot.plot(class_1.iloc[:, [0]].to_numpy(),
            class_1.iloc[:, [1]].to_numpy(), "g+")
# pyplot.plot(class_2.iloc[:, [0]].to_numpy(), class_2.iloc[:, [1]].to_numpy(), "r+")
pyplot.plot(class_3.iloc[:, [0]].to_numpy(),
            class_3.iloc[:, [1]].to_numpy(), "b+")
# pyplot.show()

class_1_vectors = class_1.iloc[:, [0, 1]].to_numpy()
class_3_vectors = class_3.iloc[:, [0, 1]].to_numpy()
class_1_y = class_1["target"].to_numpy()
class_3_y = class_3["target"].to_numpy()

print(class_3_y)

samples = np.concatenate((class_1_vectors, class_3_vectors), axis=0)
classes = np.concatenate(([1 for _ in class_1_y], [-1 for _ in class_3_y]), axis=0)

indices = [i for i in range(len(samples))]
np.random.seed(781015)
np.random.shuffle(indices)
train_indices = indices[:50]

X = samples[train_indices]
Y = classes[train_indices]

svm_optimizer = SVM()
svm_optimizer.train(X, Y)

test_indices = indices[50:]
TX = samples[test_indices]
TY = classes[test_indices]

print(svm_optimizer.score(TX, TY))


