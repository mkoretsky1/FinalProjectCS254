'''Lowell Machine Learning'''

### Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model_setup
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier



### Main ###
# Setting up
nypd = model_setup.set_up()
print(len(nypd))

# Getting rid of Staten Island
nypd_no_stat = nypd[nypd.BORO_NM != 'STATEN ISLAND']

# Splitting data
X_train, X_test, y_train, y_test = model_setup.split_data(nypd_no_stat)

# Standardizing data
X_train, X_test = model_setup.standardize(X_train, X_test)

# Trying gradient boosting
print("Gradient Boosting: \n")
gbr_cv = model_setup.gradient_boosting()
clf = OneVsRestClassifier(gbr_cv).fit(X_train, y_train)
pred = clf.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print('\n')
#create visualization for confusion matrix

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(clf,X_test,y_test, ax = ax)
plt.title('Gradient Boosting Confusion Matrix')
plt.show()


# Trying random forest
print("Random Forest: \n")
rf_cv = model_setup.random_forest()
clf = OneVsRestClassifier(rf_cv).fit(X_train, y_train)
pred = clf.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print('\n')
#create visualization for confusion matrix

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(clf,X_test,y_test, ax = ax)
plt.title('Random Forest Confusion Matrix')
plt.show()


# Trying linear regression
print('Linear Regression: \n')
lr_cv = model_setup.gradient_boosting()
clf = OneVsRestClassifier(lr_cv).fit(X_train, y_train)
pred = clf.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print('\n')

#create visualization for confusion matrix

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(clf,X_test,y_test, ax = ax)
plt.title('Linear Regression Confusion Matrix')
plt.show()




