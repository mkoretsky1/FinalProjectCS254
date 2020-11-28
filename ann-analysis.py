""" Final Project CS 254 - ANN Analysis """

import tensorflow as tf
import pandas as pd
import model_setup
from tensorflow.keras.utils import to_categorical
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
X = nypd_no_stat.drop(['BORO_NUM'], axis=1)
y = nypd_no_stat['BORO_NUM'].values

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Making categorical variables (one-hot)
num_classes = 4
y_train = to_categorical(y_train, num_classes)
