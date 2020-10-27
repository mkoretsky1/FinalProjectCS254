""" CS 254 Final Project - ML Analysis """

### Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sstats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


### Functions ###
def standardize(X_train, X_test):
    scaler = StandardScaler()
    # Fitting and transforming training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Tranforming testing data based on traning fit (prevent data leakage)
    X_test = scaler.transform(X_test)
    return X_train, X_test

### Main ###
nypd = pd.read_csv('nypd_data/nypd_10000', parse_dates=['complaint_datetime'])
nypd = nypd.dropna()
print(len(nypd))

# Getting X data
drop = ['CMPLNT_NUM','complaint_datetime','boro_BRONX','boro_BROOKLYN','boro_MANHATTAN',
        'boro_QUEENS','boro_STATEN ISLAND','Latitude','Longitude']
X = nypd.drop(drop, axis=1)

# Response variable - starting with Manhattan
y = nypd['boro_BRONX'].values

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Scaling
X_train, X_test = standardize(X_train, X_test)

# Trying a random forest classifier (should ignore NA values)
rf = RandomForestClassifier()
params = {'n_estimators':sstats.randint(10,200)}
rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=5)
rf_cv.fit(X_train, y_train)
pred = rf_cv.predict(X_test)
print(accuracy_score(y_test, pred))


# Trying an SVM classifier
sv = SVC(kernel = 'linear')
sv.fit(X_train, y_train)
svm_p = sv.predict(X_test)
print(accuracy_score(y_test,svm_p))

# Logistic Regression
log_r = LogisticRegression()
log_r.fit(X_train,y_train)
classifier = log_r.predict(X_test)
print(accuracy_score(y_test,log_r))