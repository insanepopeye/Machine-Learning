#Import required libraries 
import keras #library for neural network
import pandas as pd #loading data in table form  
import numpy as np # linear algebra

        # Importing the dataset
data=pd.read_csv('data.csv')

        #dropping the dates column
data=data.drop(['from_date','to_date','booking_created'], axis=1)

X=data.iloc[:,:15]
y=data.iloc[:,[15]]

        # Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        #creating deeplearning model
from keras.models import Sequential 
from keras.layers import Dense,Dropout 
from keras.utils import np_utils
y_train=np_utils.to_categorical(y_train,num_classes=2)
y_test=np_utils.to_categorical(y_test,num_classes=2)

model=Sequential()
model.add(Dense(50,input_dim=15,activation='hard_sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dropout(0.8))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='RMSprop',metrics=['accuracy'])

model.summary()

        #fitting the model on the data
Classifier= model.fit(X_train,y_train,validation_data=(X_test,y_test),
                      batch_size=20,epochs=5,verbose=1)

        #Checking the accuracy of the model
prediction=model.predict(X_test)
length=len(prediction)
length
y_label=np.argmax(y_test,axis=1)
y_label
predict_label=np.argmax(prediction,axis=1)
predict_label

Accuracy=np.sum(y_label==predict_label)/length * 100 
print("Accuracy of the dataset",Accuracy )
