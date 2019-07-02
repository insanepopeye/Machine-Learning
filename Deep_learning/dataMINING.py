
#Import required libraries 
import time as t
import keras #library for neural network
import pandas as pd #loading data in table form  
import numpy as np # linear algebra


data=pd.read_csv('KDDTrain_binary_class.csv')


X=data.iloc[:,4:41]
y=data.iloc[:,[41]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from keras.models import Sequential 
from keras.layers import Dense,Dropout 
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,num_classes=2)
y_test=np_utils.to_categorical(y_test,num_classes=2)

model=Sequential()
model.add(Dense(50,input_dim=37,activation='hard_sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='SGD',metrics=['accuracy'])

model.summary()


model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=3,verbose=1)


prediction=model.predict(X_test)
length=len(prediction)
length
y_label=np.argmax(y_test,axis=1)
y_label
predict_label=np.argmax(prediction,axis=1)
predict_label

Accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",Accuracy )

