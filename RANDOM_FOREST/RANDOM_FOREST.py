# RNADOM FOREST

# Importing the libraries
#import numpy as np

import pandas as pd
import time as t
# Importing the dataset
dataset = pd.read_csv('GAMMA.csv')
X = dataset.iloc[:, :11].values
y = dataset.iloc[:, [11]].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( criterion='entropy',random_state = 0)

from sklearn import tree
classifier = tree.DecisionTreeClassifier(criterion='entropy')

t0= t.time()
t0
classifier.fit(X_train, y_train)
t1 = t.time()
t1
Model_build = t1 - t0
Model_build


# Predicting the Test set results
t2 = t.time()
y_pred = classifier.predict(X_test)
t3 = t.time()
Test_time = t3 - t2
Test_time


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

#Model's Accuracy
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)
accuracy

#MODEL PRECISION & RECALL
from sklearn.metrics import precision_recall_fscore_support
precision = precision_recall_fscore_support(y_test,y_pred, average='macro')
precision

tree.export_graphviz(classifier,out_file='tree.dot') 
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])