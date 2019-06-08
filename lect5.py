#Write a own classifier
import random
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)
    
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train


    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions 

    def closest(self, row):
        best_fit = euc(row, self.X_train[0])
        best_index = 0
        for i in range (1, len(self.X_train)):
            dist = euc(row , self.X_train[i])
            if dist < best_fit:
                best_fit = dist
                best_index = i
        return self.y_train[best_index]
        

 # first goal is to make pipleline working
 #        
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
my_classifier = ScrappyKNN()
my_classifier.fit(X_train,y_train)

# predict for test data
prediction = my_classifier.predict(X_test)
print(prediction)

# Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction))
