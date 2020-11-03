""" CS 254 Final Project - ML Analysis """

### Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sstats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
print(len(nypd.columns))

# Response variable - starting with Manhattan
y = nypd['BORO_NM'].values

# Resampling
ros = SMOTE(random_state=0)
X_resample, y_resample = ros.fit_resample(X, y)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Scaling
X_train, X_test = standardize(X_train, X_test)

rf = RandomForestClassifier()
svc = SVC()
params = {'n_estimators':sstats.randint(10,200)}
rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=5)
clf = OneVsRestClassifier(svc).fit(X_train, y_train)
pred = clf.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))