import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np

#matplotlib inline
 #IMORTING THE DATASET.
dataset = pd.read_csv('KDDCup99_binary_classlable.csv')
X = dataset.iloc[:, :38]
y = dataset.iloc[:, [38]]


#SPLITTING OF THE TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


#CHANGING CATEGORICAL VALUES TO NUMERIC VALUES.
'''from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
X[:,1] = encode.fit_transform(X[:,1])
X[:,2] = encode.fit_transform(X[:,2])
X[:,3] = encode.fit_transform(X[:,3])'''


#PLOTING THE CLASS LABELS
pd.value_counts(dataset['class_label']).plot.bar()
plt.title('IDS class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
dataset['class_label'].value_counts()


#SHAPE VIEWING
X = np.array(dataset.iloc[:, dataset.columns != 'class_label'])
y = np.array(dataset.iloc[:, dataset.columns == 'class_label'])
print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))


print("Number transactions X_train dataset:", X_train.shape)
print("Number transactions y_train dataset:", y_train.shape)
print("Number transactions X_test dataset:", X_test.shape)
print("Number transactions y_test dataset:", y_test.shape)



#APPLYING SMOTE OVERSAMPLING
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


features = pd.DataFrame(X_train_res)
features.to_csv('features.csv')

classlable = pd.DataFrame(y_train_res)
classlable.to_csv('classlable.csv')



A = [features,classlable] 
SMOTE_TRAIN_DATA = pd.DataFrame(A)