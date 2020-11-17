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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import TomekLinks



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

# Joining training data back together (for over and undersampling)
X = pd.DataFrame(X_train)

# Adding borough name variable back in
X['BORO_NM'] = y_train

# Separating categories
bronx = X[X.BORO_NM == 'BRONX']
brooklyn = X[X.BORO_NM == 'BROOKLYN']
manhattan = X[X.BORO_NM == 'MANHATTAN']
queens = X[X.BORO_NM == 'QUEENS']
staten = X[X.BORO_NM=='STATEN ISLAND']
other = X[X.BORO_NM != 'STATEN ISLAND']

# Getting the lengths
print(len(bronx), len(brooklyn), len(manhattan), len(queens), len(staten), len(other))

# # Oversampling Staten Island
# staten_over = resample(staten,replace=True,n_samples=len(manhattan),random_state=10)
# print(len(staten_over))

# # Oversampling Queens
# queens_over = resample(queens,replace=True,n_samples=len(manhattan), random_state=10)
# print(len(queens_over))

# # Undersampling Brooklyn
# brooklyn_under = resample(brooklyn,replace=False,n_samples=len(manhattan), random_state=10)
# print(len(brooklyn_under))

# # Reseparating training data
# X_train = pd.concat([other, staten_over])
# y_train = X_train.BORO_NM
# X_train = X_train.drop(['BORO_NM'],axis=1)

# # Oversampling using SMOTE
# sm = SMOTE(random_state=10)
# X_train, y_train = sm.fit_resample(X_train, y_train)


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
fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(transformed_data[:, 0][class_0], transformed_data[:, 1][class_0], label = "Bronx")
plt.scatter(transformed_data[:, 0][class_1], transformed_data[:, 1][class_1], label = "Brooklyn")
plt.scatter(transformed_data[:, 0][class_2], transformed_data[:, 1][class_2], label = "Queens" )
plt.scatter(transformed_data[:, 0][class_3], transformed_data[:, 1][class_3], label = "Staten Island ")
plt.scatter(transformed_data[:, 0][class_4], transformed_data[:, 1][class_4], label = "Manhatten")
plt.legend()
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


#Mat did random forest below are the results

#Bronx got an accuracy of 0.777 using random forest

#Manhattan got an accuracy of 0.767 using random forest

#Brooklyn got an accuracy of 0.702401 using random forest

#Queens got an accuracy of 0.816211 using random forest

#Staten Island got an accuracy of 0.96 using random forest (not a good measurement due to skew)

#if we try to predict the boro in one model we get
#a accuracy score of 0.398 using random forest (maybe should use a different accuracy measurement due to skew)

#building a KNN neighbors model
KNN = KNeighborsClassifier()
params = {'n_neighbors':[5,10,15,20,25,30,35],
          'weights':['uniform', 'distance']}
KNN_cv = RandomizedSearchCV(estimator=KNN, param_distributions=params, n_iter=5,scoring='f1_weighted')
clf = OneVsRestClassifier(KNN_cv).fit(X_train, y_train)
pred = clf.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

#create a confusion matrix visualization

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(clf,X_test,y_test, ax = ax)
plt.title('K Nearest Neighbors Confusion Matrix')
plt.show()

# Trying gradient boosting
gbr = GradientBoostingClassifier()
params = {'n_estimators':[25,50,75,100,150,200],
          'loss':['deviance','exponential'],}
gbr_cv = RandomizedSearchCV(estimator=gbr, param_distributions=params, n_iter=5,scoring='f1_weighted')
clf = OneVsRestClassifier(gbr_cv).fit(X_train, y_train)
pred = clf.predict(X_test)
print(pd.DataFrame(confusion_matrix(y_test, pred)))
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))


#create visualization for confusion matrix

fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(clf,X_test,y_test, ax = ax)
plt.title('Gradient Boosting Confusion Matrix')
plt.show()



