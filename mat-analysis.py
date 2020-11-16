""" CS 254 Final Project - ML Analysis """

### Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sstats
import model_setup
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

### Main ###
# Setting up
nypd = model_setup.set_up()
print(len(nypd))

# Splitting data
X_train, X_test, y_train, y_test = model_setup.split_data(nypd)

# Standardizing data
X_train, X_test = model_setup.standardize(X_train, X_test)

# Joining training data back together (for over and undersampling)
X = pd.DataFrame(X_train)

# Adding borough name variable back in
X['BORO_NM'] = y_train

# Separating categories
bronx = X[X.BORO_NM=='BRONX']
brooklyn = X[X.BORO_NM=='BROOKLYN']
manhattan = X[X.BORO_NM=='MANHATTAN']
queens = X[X.BORO_NM=='QUEENS']
staten = X[X.BORO_NM=='STATEN ISLAND']
other = X[X.BORO_NM!='STATEN ISLAND']

# Getting the lengths 
print(len(bronx), len(brooklyn), len(manhattan), len(queens), len(staten), len(other))

# Oversampling Staten Island
staten_over = resample(staten,replace=True,n_samples=len(manhattan),random_state=10)
print(len(staten_over))

# Reseparating training data
X_train = pd.concat([other, staten_over])
y_train = X_train.BORO_NM
X_train = X_train.drop(['BORO_NM'],axis=1)

# Random forest
rf_cv = model_setup.random_forest()

# Checking cross-validation results
## Comment these two lines out when specifying scoring metric in cross-validation
## Note: seeing best performance for random forest when auc is used as the metric
rf_cv.fit(X_train, y_train)
print(rf_cv.best_params_)

# One vs rest classifier for testing
clf = OneVsRestClassifier(estimator=rf_cv)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Confusion matrix, classification report, accuracy
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))