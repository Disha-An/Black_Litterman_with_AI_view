import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 35, kernel_initializer = "uniform", input_dim = 48))
    classifier.add(Dense(units = 35, kernel_initializer = "uniform"))
    classifier.add(Dense(units = 35, kernel_initializer = "uniform"))
    classifier.add(Dense(units = 35, kernel_initializer = "uniform"))
    classifier.add(Dense(units = 35, kernel_initializer = "uniform"))
    classifier.add(Dense(units = 1, kernel_initializer = "uniform"))
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
    return classifier

train_start_date = '2010-01-01'
train_end_date = '2016-12-31'
test_start_date = '2017-01-01'
test_end_date = '2017-10-31'

# ETFs list : IJK, IVV, IWP, IWV, IYY, LQD, VOT
## LQD
all_data = pd.read_csv('LQD.csv') # This is the file generated from ANN's inputs data preprocessing sode
mask = (all_data['Date'] > train_start_date) & (all_data['Date'] <= train_end_date)
train_dataset = all_data.loc[mask]
X_train = train_dataset.iloc[:, 1:49].values
y_train = []
l = len(train_dataset)
for i in range(l-19):
    y_train.append(sum(train_dataset.iloc[i:20+i, 20:21].values))
y_train = np.array(y_train)
X_train = X_train[:l-19,:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Training the data
classifier = build_classifier()
classifier.fit(X_train, y_train, batch_size = 49, epochs = 500)

# Predicting the result
mask = (all_data['Date'] > test_start_date) & (all_data['Date'] <= test_end_date)
test_dataset = all_data.loc[mask]
X_test = test_dataset.iloc[:, 1:49].values
y_test = []
l = len(test_dataset)
for i in range(l-19):
    y_test.append(sum(test_dataset.iloc[i:20+i, 20:21].values))
y_test = np.array(y_test)
X_test = X_test[:l-19,:]
sc = StandardScaler()
X_test = sc.fit_transform(X_test)
y_pred = classifier.predict(X_test)