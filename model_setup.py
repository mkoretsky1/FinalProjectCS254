""" CS 254 Final Project - Model Setup """

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def set_up():
    nypd = pd.read_csv('nypd_data/nypd_10000.csv', parse_dates=['complaint_datetime'])
    # Getting X data
    # Variables to drop regardless of the analysis
    drop_always = ['CMPLNT_NUM','SUSP_RACE','SUSP_SEX','VIC_SEX','complaint_datetime','Unnamed: 0']
    # Variables to drop when performing classification for location
    drop_for_location_analysis = ['Latitude','Longitude','boro_BRONX','boro_BROOKLYN','boro_MANHATTAN',
                                  'boro_QUEENS','boro_STATEN ISLAND']
    # Creating one list of variables to drop - Edit this line based on analysis being performed
    drop = drop_always + drop_for_location_analysis
    nypd = nypd.drop(drop, axis=1)
    nypd = nypd.dropna(axis=0)
    return nypd
    
def split_data(nypd):
    # X data
    X = nypd.drop(['BORO_NM'], axis=1)
    # Response variable
    y = nypd['BORO_NM'].values
    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=10)
    return X_train, X_test, y_train, y_test
    
def standardize(X_train, X_test):
    scaler = StandardScaler()
    # Fitting and transforming training data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    # Tranforming testing data based on traning fit (prevent data leakage)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def random_forest():
    rf = RandomForestClassifier(class_weight='balanced')
    params = {'n_estimators':[25,50,75,100,150,200], 'max_depth':[2,3,4,5,6,7,8,9,10],
              'max_features':['sqrt','log2',25,50,75,100]}
    rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=5)
    return rf_cv


    
    