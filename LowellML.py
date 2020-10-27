'''Lowell Machine Learning'''

### Imports ###
import numpy as np
import pandas as pd
import scipy.stats as sstats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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
drop = ['CMPLNT_NUM','complaint_datetime','BORO_NM','Latitude','Longitude']
X = nypd.drop(drop, axis=1)

# Response variable - starting with Manhattan
y = nypd['BORO_NM'].values

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Scaling
X_train, X_test = standardize(X_train, X_test)

#lets try looking at the split of the boros with PCA
#transform the data to two components
clf = PCA(n_components=2)
transformed_data = clf.fit_transform(X_train)

#create a scatter plot with different colors for different clases of data-points
class_0 = np.where(y_train == "BRONX")
class_1 = np.where(y_train == "BROOKLYN")
class_2 = np.where(y_train == "QUEENS")
class_3 = np.where(y_train == "STATEN ISLAND")
class_4 = np.where(y_train == "MANHATTAN")

# plotting of transformed data by class
plt.scatter(transformed_data[:, 0][class_0], transformed_data[:, 1][class_0])
plt.scatter(transformed_data[:, 0][class_1], transformed_data[:, 1][class_1])
plt.scatter(transformed_data[:, 0][class_2], transformed_data[:, 1][class_2])
plt.scatter(transformed_data[:, 0][class_3], transformed_data[:, 1][class_3])
plt.scatter(transformed_data[:, 0][class_4], transformed_data[:, 1][class_4])


# Trying a random forest classifier (should ignore NA values)
rf = RandomForestClassifier()
params = {'n_estimators':sstats.randint(10,200)}
rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=params, n_iter=5)
rf_cv.fit(X_train, y_train)
pred = rf_cv.predict(X_test)
print(accuracy_score(y_test, pred))

#Bronx got an accuracy of 0.777

#Manhattan got an accuracy of 0.767

#Brooklyn got an accuracy of 0.702401

#Queens got an accuracy of 0.816211

#Staten Island got an accuracy of 0.96

#if we try to predict the boro in one model we get
#a very bad accuracy score of 0.398

#lets look at the confusion matrix
confRF = confusion_matrix(y_test, pred)

conflogpd = pd.DataFrame(confRF)
print(conflogpd)

