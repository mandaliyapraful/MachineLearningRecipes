# import dataset
# Train the classifer
# Predict the label for new flower
# Visualize the tree

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()

print(iris.data[0])
print(iris.target[0])

for i in range(len(iris.target)):
    print("Example %d: labels %s, feature %s" %(i,iris.target[i],iris.data[i]))

test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# call classifer
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data,train_target)

print(test_target)
print(clf.predict(test_data))

# viz code 
from sklearn.externals.six import StringIO
import pydot

""" dot_data = StringIO()
tree.export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf') """

