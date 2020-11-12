""" CS 254 Final Project - Feature Importance """


### Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib as plt
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

# Getting X data
# Variables to drop regardless of the analysis
drop_always = ['CMPLNT_NUM','SUSP_RACE','SUSP_SEX','VIC_SEX','complaint_datetime','Unnamed: 0']
# Variables to drop when performing classification for location
drop_for_location_analysis = ['Latitude','Longitude','BORO_NM']
# Variables to drop for brooklyn one vs all
drop_brooklyn = ['boro_BRONX','boro_MANHATTAN',
                              'boro_QUEENS','boro_STATEN ISLAND']
# Variables to drop for queens one vs all
drop_queens = ['boro_BRONX','boro_BROOKLYN','boro_MANHATTAN',
                              'boro_STATEN ISLAND']
# Variables to drop for staten island one vs all
drop_staten_island = ['boro_BRONX','boro_BROOKLYN','boro_MANHATTAN',
                              'boro_QUEENS']
# Variables to drop for bronx one vs all
drop_bronx = ['boro_BROOKLYN','boro_MANHATTAN',
                              'boro_QUEENS','boro_STATEN ISLAND']
# Variables to drop for manhatten one vs all
drop_manhattan = ['boro_BRONX','boro_BROOKLYN',
                              'boro_QUEENS','boro_STATEN ISLAND']

# Creating one list of variables to drop - Edit this line based on analysis being performed
dropbrook = drop_always + drop_for_location_analysis + drop_brooklyn
dropbronx = drop_always + drop_for_location_analysis + drop_bronx
dropqueen = drop_always + drop_for_location_analysis + drop_queens
dropman = drop_always + drop_for_location_analysis + drop_manhattan
dropstat = drop_always + drop_for_location_analysis + drop_staten_island

drop = [dropbrook, dropbronx, dropqueen, dropman, dropstat]
boro = ['boro_BROOKLYN','boro_BRONX','boro_QUEENS','boro_MANHATTAN', 'boro_STATEN ISLAND']

for i in range(5):

    #get the data
    nypd = pd.read_csv('nypd_data/nypd_10000.csv', parse_dates=['complaint_datetime'])
    nypd = nypd.drop(drop[i], axis=1)

    nypd = nypd.dropna(axis=0)
    print(len(nypd))

    X = nypd.drop([boro[i]], axis=1)

    # Response variable
    y = nypd[boro[i]]

    # Oversampling
    ros = SMOTE(random_state=0)

    X_resample, y_resample = ros.fit_resample(X, y)

    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

    # Scaling
    X_train, X_test = standardize(X_train, X_test)

    rf = RandomForestClassifier(class_weight='balanced')
    params = {'n_estimators':[25,50,75,100,150,200], 'max_depth':[2,3,4,5,6,7,8,9,10],
          'max_features':['sqrt','log2',25,50,75,100]}
    rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=5, scoring='f1_weighted')
    rf_cv.fit(X_train, y_train)

    pred = rf_cv.predict(X_test)
    print(pd.DataFrame(confusion_matrix(y_test, pred)))
    print(classification_report(y_test, pred))
    print(accuracy_score(y_test, pred))

    feature_importances = pd.DataFrame(rf_cv.best_estimator_.feature_importances_, index = X.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importances.head(10))
