""" CS 254 Final Project - Feature Importance """


### Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib as plt
import model_setup
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier


#get the data
nypd = model_setup.set_up()
print(len(nypd))

# Getting rid of Staten Island
nypd_no_stat = nypd[nypd.BORO_NM != 'STATEN ISLAND']
#nypd_no_stat = nypd.drop(["boro_BRONX","boro_BROOKLYN","boro_QUEENS","boro_MANHATTAN", "boro_STATEN ISLAND"], axis = 1)

# Splitting data
X_train, X_test, y_train, y_test = model_setup.split_data(nypd_no_stat)
X_train, X_test = model_setup.standardize(X_train, X_test)


X = nypd_no_stat.drop(['BORO_NM'], axis=1)



# Random forest
# rf_cv = model_setup.random_forest()
# rf_cv.fit(X_train, y_train)
#
# # One vs rest classifier for testing
# clf_rf = OneVsRestClassifier(estimator=rf_cv)
# clf_rf.fit(X_train, y_train)
# pred = clf_rf.predict(X_test)
# print(classification_report(y_test, pred))

#do the same for gradient boosting
gbr_cv = model_setup.gradient_boosting()
gbr_cv.fit(X_train, y_train)

# One vs rest classifier for testing
clf_gbr = OneVsRestClassifier(estimator=gbr_cv)
clf_gbr.fit(X_train, y_train)
pred = clf_gbr.predict(X_test)
print(classification_report(y_test, pred))

#do the same for logistic regression
# lr_cv = model_setup.log_reg()
# lr_cv.fit(X_train, y_train)
#
# # One vs rest classifier for testing
# clf_lr = OneVsRestClassifier(estimator=lr_cv)
# clf_lr.fit(X_train, y_train)
# pred = clf_lr.predict(X_test)
# print(classification_report(y_test, pred))

for i in range(4):
    # print which method it is

    # print("Random Forest \n")
    # print("Feature importance for the ", clf_rf.classes_[i], "\n")
    #
    # feature_importances_rf = pd.DataFrame(clf_rf.estimators_[i].best_estimator_.feature_importances_, index=X_test.columns,
    #                                       columns=["importance"]).sort_values('importance', ascending = False)
    #
    # print(feature_importances_rf.head(10))
    #
    # plt.pyplot.barh(feature_importances_rf.head(10).index, feature_importances_rf.head(10)['importance'])
    # plt.pyplot.title('Random Forest feature importance for ' + clf_rf.classes_[i])
    # plt.pyplot.xlabel('importance')
    # plt.pyplot.show()

    # print which method it is

    print("Gradient Boosting \n")
    print("Feature importance for the ", clf_gbr.classes_[i], "\n")

    feature_importances_gbr = pd.DataFrame(clf_gbr.estimators_[i].best_estimator_.feature_importances_,
                                          index=X.columns,
                                          columns=["importance"]).sort_values('importance', ascending=False)

    print(feature_importances_gbr.head(10))


    plt.pyplot.barh(feature_importances_gbr.head(10).index, feature_importances_gbr.head(10)['importance'])
    plt.pyplot.title('Gradient Boosting feature importance for ' + clf_gbr.classes_[i])
    plt.pyplot.xlabel('importance')
    plt.pyplot.show()

    # print which method it is

    # print("Logistic Regression \n")
    # print("Feature importance for ", clf_lr.classes_[i], "\n")
    #
    # feature_importances_lr = pd.DataFrame(clf_lr.estimators_[i].best_estimator_.feature_importances_,
    #                                       index=X_test.columns,
    #                                       columns=["importance"]).sort_values('importance', ascending=False)
    #
    # print(feature_importances_lr.head(10))
    #
    # plt.pyplot.barh(feature_importances_lr.head(10).index, feature_importances_lr.head(10)['importance'])
    # plt.pyplot.title('Logistic Regression feature importance for ' + clf_rf.classes_[i])
    # plt.pyplot.xlabel('importance')
    # plt.pyplot.show()


















