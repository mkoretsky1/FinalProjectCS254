# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:42:02 2020

@author: raytr
"""


""" Final Project CS 254 - ANN Analysis """

import tensorflow as tf
import pandas as pd
import model_setup
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt


### Main ###
# Setting up
nypd = model_setup.set_up()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# Mapping boroughs for one-hot encoding and dropping the name variable
nypd['BORO_NUM'] = nypd['BORO_NM'].map({'BRONX':0,'BROOKLYN':1,'MANHATTAN':2,'QUEENS':3,'STATEN ISLAND':4})
nypd = nypd.drop(['BORO_NM'], axis=1)

# Getting rid of Staten Island
nypd_no_stat = nypd[nypd.BORO_NUM != 4]

# Getting X and y data
y = nypd_no_stat['BORO_NUM'].values
X = nypd_no_stat.drop(['BORO_NUM'], axis=1)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

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
model.add(layers.Dense(100, input_shape=(539,), activation='tanh'))
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.LeakyReLU())
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
opt = optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',f1_m,precision_m, recall_m])
model.fit(X_train, y_train, epochs=10, batch_size=16)
model.evaluate(X_test, y_test)


# fit the model
history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, verbose=0)

# evaluate the model
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
