# DECISION TREE J48

# Importing the libraries
#import numpy as np

import pandas as pd
import time as t
# Importing the dataset
dataset = pd.read_csv('KDD-CUP99-SMOTE-TRAINING.csv')
X = dataset.iloc[:, :38]
y = dataset.iloc[:, [38]]

Test_dataset = pd.read_csv('KDD-CUP99-SMOTE-TESTNG.csv')
X_test = Test_dataset.iloc[:, :38]
y_test = Test_dataset.iloc[:, [38]]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

t0= t.time()
t0
classifier.fit(X, y)
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
