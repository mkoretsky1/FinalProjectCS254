""" CS 254 Final Project - ML Analysis """

### Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sstats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report


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
nypd = pd.read_csv('nypd_data/nypd_10000.csv', parse_dates=['complaint_datetime'])
nypd = nypd.dropna()
print(len(nypd))

# Getting X data
# Variables to drop regardless of the analysis
drop_always = ['CMPLNT_NUM','complaint_datetime','BORO_NM','time_of_day','season','tod_afternoon',
               'tod_morning','tod_night','season_fall','season_spring','season_summer','season_winter']
# Variables to drop when performing classification for location
drop_for_location_analysis = ['Latitude','Longitude','boro_BRONX','boro_BROOKLYN','boro_MANHATTAN',
                              'boro_QUEENS','boro_STATEN ISLAND']
# Variables to drop when performing classification for tod
drop_for_tod_analysis = ['hour','minute']
# Variables to drop when performing classification for season
drop_for_season_analysis = ['month']
# Creating one list of variables to drop - Edit this line based on analysis being performed
drop = drop_always + drop_for_location_analysis
X = nypd.drop(drop, axis=1)


# Response variable - starting with Manhattan
y = nypd['BORO_NM'].values

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

#lets get the full report
print(classification_report(y_test, pred))

#create visualization for confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(rf_cv,X_test,y_test, ax = ax)
plt.title('Random Forest Confusion Matrix')
plt.show()

# Trying an SVM classifier
sv = SVC(kernel = 'linear')
sv.fit(X_train, y_train)
svm_p = sv.predict(X_test)
print(accuracy_score(y_test,svm_p))

#lets get the full report
print(classification_report(y_test, svm_p))

#create visualization for confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(sv,X_test,y_test, ax = ax)
plt.title('SVM Confusion Matrix')
plt.show()

# Logistic Regression
log_r = LogisticRegression(max_iter=100000)
log_r.fit(X_train,y_train)
classifier = log_r.predict(X_test)
print(accuracy_score(y_test,classifier))

#lets get the full report
print(classification_report(y_test, classifier))

#create visualization for confusion matrix
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(log_r,X_test,y_test, ax = ax)
plt.title('Logistic Regression Confusion Matrix')
plt.show()