""" CS 254 Final Project - Model Setup """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

def set_up():
    nypd = pd.read_csv('nypd_data/nypd_100000.csv', parse_dates=['complaint_datetime'])
    # Getting X data
    # Variables to drop regardless of the analysis
    drop_always = ['CMPLNT_NUM','SUSP_RACE','SUSP_SEX','VIC_SEX','complaint_datetime','Unnamed: 0','Unnamed: 0.1']
    # Variables to drop when performing classification for location
    drop_for_location_analysis = ['Latitude','Longitude']
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

def k_nearest_neighbors():
    KNN = KNeighborsClassifier()
    params = {'n_neighbors': [5, 10, 15, 20, 25, 30, 35],
              'weights': ['uniform', 'distance']}
    KNN_cv = RandomizedSearchCV(estimator=KNN, param_distributions=params, n_iter=5, scoring='f1_weighted')
    return KNN_cv

def gradient_boosting():
    gbc = GradientBoostingClassifier()
    params = {'n_estimators': [25, 50, 75, 100, 150, 200],
              'loss': ['deviance']}
    gbc_cv = RandomizedSearchCV(estimator=gbc, param_distributions=params, n_iter=5, scoring='f1_weighted')
    return gbc_cv

def support_vector():
    svc = SVC()
    params = {'C':[0.01,0.05,0.1,0.15,1.0], 'gamma':[1e-5,0.001,0.01,0.1,1,10]}
    svc_cv = RandomizedSearchCV(estimator=svc, param_distributions=params, n_iter=5, scoring='f1_weighted')
    return svc_cv

def log_reg():
    log_reg = LogisticRegressionCV(Cs=[0.001,0.01,0.05,0.1,0.5], random_state=10, cv=5, 
                                   scoring='f1_weighted')
    return log_reg
    
    
    
    