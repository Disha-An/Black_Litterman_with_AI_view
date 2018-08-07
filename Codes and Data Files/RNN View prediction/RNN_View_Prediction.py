import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

train_start = '2010-01-01'
train_end = '2016-12-31'
test_start = '2016-07-13'
test_end = '2017-10-31'

def build_regressor():
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return regressor

# ETFs list : IJK, IVV, IWP, IWV, IYY, LQD, VOT
### LQD ###
data = pd.read_csv('LQD.csv') # Preprocessed daily log-returns data file
mask = (data['Date'] >= train_start) & (data['Date'] <= train_end)
train_dataset = data.loc[mask]
train_dataset = train_dataset.iloc[:,1:2].values
sc = StandardScaler()
scaled_train = sc.fit_transform(train_dataset)
X_train = []
y_train = []
for i in range(len(scaled_train)-120):
    X_train.append(scaled_train[i:i+120,0])
    y_train.append(sum(train_dataset[i+120:i+140]))
y_train = sc.fit_transform(y_train)
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = build_regressor()
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

mask = (data['Date'] >= test_start) & (data['Date'] <= test_end)
test_dataset = data.loc[mask]
test_dataset = test_dataset.iloc[:,1:2].values
scaled_test = sc.fit_transform(test_dataset)
X_test = []
y_test = []
for i in range(len(scaled_test)-120):
    X_test.append(scaled_test[i:i+120,0])
    y_test.append(sum(test_dataset[i+120:i+140]))
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_pred = regressor.predict(X_test)
y_pred = sc.inverse_transform(y_pred)