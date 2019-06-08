#Write a Pipleline
from sklearn import datasets
iris = datasets.load_iris()
# set the X , y 
X = iris.data
y = iris.target

# import train_test_split from model selection --crossvalidation
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.5)

# train the classifer
from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()
#my_classifier.fit(X_train,y_train)

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train,y_train)

# predict for test data
prediction = my_classifier.predict(X_test)
print(prediction)

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction))
