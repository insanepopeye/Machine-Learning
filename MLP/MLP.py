# MULTILAYER PERCEPTRON

# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import time as t

# Importing the dataset
dataset = pd.read_csv('NSL_NO_CATEGORICAL_COV_800.csv')
X = dataset.iloc[:, :15]
y = dataset.iloc[:, [15]]

#CTAGORICAL TO NUMERICAL
df_with_dummies = pd.get_dummies(X)
df_with_dummies
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting MLP to the Training set
from sklearn.neural_network import MLPClassifier
classifier =  MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)

t0= t.time()
t0
classifier.fit(X_train, y_train)
t1 = t.time()
t1
T_Final = t1 - t0
T_Final


# Predicting the Test set results
T2 = t.time()
y_pred = classifier.predict(X_test)
T3= t.time()
FINAL = T3 - T2
FINAL

TOTAL_TIME = T_Final + FINAL 
TOTAL_TIME 
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






#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(500, 100), random_state=1)
#classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 10), random_state=0)
'''classifier=MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)'''

#print "training time:", round(time()-t0, 3), "s" # the time would be round to 3 decimal in seconds
'''MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(5, 2),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)'''

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)