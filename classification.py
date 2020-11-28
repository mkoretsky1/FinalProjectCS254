""" CS 254 Final Project - Classification Analysis """

### Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model_setup
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier

### Functions ###

### Main ###
# Setting up model
nypd = model_setup.set_up()
print(len(nypd))

# Dropping Staten Island (separate df for easy use)
nypd_no_staten = nypd[nypd.BORO_NM != 'STATEN ISLAND']
print(len(nypd_no_staten))

# Splitting and standardizing data
X_train, X_test, y_train, y_test = model_setup.split_data(nypd)
X_train, X_test = model_setup.standardize(X_train, X_test)

# Getting model
model = model_setup.random_forest()

# Fitting model
model.fit(X_train, y_train)
# Predicting
pred = model.predict(X_test)
# Metrics and visualizations
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
plot_confusion_matrix(model, X_test, y_test)
plt.show()

# Fitting OneVsRest model
ovr = OneVsRestClassifier(estimator=model)
ovr.fit(X_train, y_train)
# Predicting
pred = ovr.predict(X_test)
# Metrics and visualizations
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
plot_confusion_matrix(ovr, X_test, y_test)
plt.show()

# Best parameters from OvR (needs to be commented out for log_reg)
for i in range(len(ovr.estimators_)):
    print(ovr.estimators_[i].best_params_)
    
# Looking at feature importances for the ovr models

