# SVM-C

# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import time as t

# Importing the dataset
dataset = pd.read_csv('KDDTrain+.arff.csv')
X = dataset.iloc[:, 4:41]
y = dataset.iloc[:, [41]]

#CTAGORICAL TO NUMERICAL
df_with_dummies = pd.get_dummies(X)
df_with_dummies
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting MLP to the Training set
from sklearn.svm import SVC
classifier = SVC(gamma='auto')
classifier = SVC(C=1.0, kernel='rbf',gamma='auto', degree=3, gamma='auto_deprecated',
                            coef0=0.0, shrinking=True, probability=False, tol=0.001,
                            cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                            decision_function_shape='ovr', random_state=None)
t0= t.time()
t0
classifier.fit(X_train, y_train)
t1 = t.time()
t1
Final = t1 - t0
Final


# Predicting the Test set results
y_pred = classifier.predict(X_test)

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


