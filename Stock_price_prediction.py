# importing all the necessary libraries

import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error

# Preprocessing the dataframe

df = pd.read_csv("/content/drive/MyDrive/NLP_Datasets/AAPL.csv")
print(f"columns in the df are {df.columns}")
print(f"shape of the df is {df.shape}")

df1= df.reset_index()['close']


#Applying minmaxscaler to scale the values to a feature range 0 to 1 

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


print(f"shape of the df1 after converting to numpy array  is:{df1.shape}")

training_size = int(len(df1)*0.70)
testing_size = len(df1)-training_size

print(f"training_size is :{training_size}")
print(f"testing_size is :{testing_size}")


#Splitting the Dataset inti train and test

train_data = df1[0:training_size,:]
test_data = df1[training_size:len(df1),:1]

print(f"shape of the training data is {train_data.shape}")
print(f"shape of the testing data {test_data.shape}")

#Defining a function for creating the dataset with the features.

def create_dataset(data,timestep=1):
   print(data.shape)
   print(len(data))
   dataX,dataY =[],[]
   for index in range(len(data)-timestep-1):
     dataX.append(data[index:(index+timestep),0])
     dataY.append(data[index+timestep,0])
   return np.array(dataX),np.array(dataY)

X_train,y_train = create_dataset(train_data,timestep=100)
X_test,y_test  = create_dataset(test_data,timestep=100)

print(f"X_train.shape is {X_train.shape}")
print(f"Y_train.shape is {Y_train.shape}")

print(f"X_test.shape is {X_test.shape}")
print(f"y_test.shape is {y_test.shape}")
print("After reshaping the X_train and X_test")

X_train =X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
print(f"X_train.shape is {X_train.shape}")
print(f"X_test.shape is {X_test.shape}")

#Building the model

model = Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mean_squared_error")
model.summary()

# training the model

model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)

#testing the model

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

print(math.sqrt(mean_squared_error(y_train,train_predict)))
print(math.sqrt(mean_squared_error(y_test,test_predict)))
