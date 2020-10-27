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
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
plt.show()


#Mat did random forest below are the results

#Bronx got an accuracy of 0.777 using random forest

#Manhattan got an accuracy of 0.767 using random forest

#Brooklyn got an accuracy of 0.702401 using random forest

#Queens got an accuracy of 0.816211 using random forest

#Staten Island got an accuracy of 0.96 using random fores

#if we try to predict the boro in one model we get
#a very bad accuracy score of 0.398 using random forest

#building a KNN neighbors model
#for i in range(1,105,2):

KNN = KNeighborsClassifier(n_neighbors=35)
KNN.fit(X_train, y_train)
pred = KNN.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, pred), " for k = ",35)

#seems to reach a max around k = 35

#lets look at the confusion matrix
confKNN = confusion_matrix(y_test, pred)

confKNNpd = pd.DataFrame(confKNN)
print(confKNNpd)





