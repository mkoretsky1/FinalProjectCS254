""" Final Project CS 254 - ANN Analysis """

import tensorflow as tf
import pandas as pd
import model_setup
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split

### Main ###
# Setting up
nypd = model_setup.set_up()

# Mapping boroughs for one-hot encoding and dropping the name variable
nypd['BORO_NUM'] = nypd['BORO_NM'].map({'BRONX':0,'BROOKLYN':1,'MANHATTAN':2,'QUEENS':3,'STATEN ISLAND':4})
nypd = nypd.drop(['BORO_NM'], axis=1)

# Getting rid of Staten Island
nypd_no_stat = nypd[nypd.BORO_NUM != 4]

# Getting X and y data
y = nypd_no_stat['BORO_NUM'].values
X = nypd_no_stat.drop(['BORO_NUM'], axis=1)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=10)

# Standardizing
X_train, X_test = model_setup.standardize(X_train, X_test)

# One-hot encoding
num_classes = 4
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Checking shapes and one-hot enocoding
print(X_train.shape)
print(y_train.shape)
print(y_train[0])
print()
print(X_test.shape)
print(y_test.shape)
print(y_test[0])

# MLP model
model = Sequential()
model.add(layers.Dense(500, input_shape=(471,), activation='tanh'))
model.add(layers.Dense(500, input_shape=(471,), activation='relu'))
model.add(layers.Dense(500, input_shape=(471,), activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
opt = optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy','Precision'])
model.fit(X_train, y_train, epochs=10, batch_size=1000, validation_split=0.1)
model.evaluate(X_test, y_test)
