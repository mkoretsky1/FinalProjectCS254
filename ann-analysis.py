""" Final Project CS 254 - ANN Analysis """

import tensorflow as tf
import pandas as pd
import model_setup
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

### Functions ###
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

def plot_learning_curve(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

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

# MLP model #1
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
history = model.fit(X_train, y_train, validation_split=0.3, epochs=10, verbose=0)
loss, accuracy, f1_score, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print(f1_score)
plot_learning_curve(history)

# # MLP model #2 
# model = Sequential()
# model.add(layers.LeakyReLU(32, input_shape=(471,)))
# model.add(layers.LeakyReLU(64))
# model.add(layers.LeakyReLU())
# model.add(layers.Dense(4, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','Precision','Recall'])
# hist = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=100)
# loss, accuracy, precision, recall = model.evaluate(X_test, y_test)
# f1 = 2*((precision*recall)/(precision+recall))
# print(f1)
# plot_learning_curve(hist)
