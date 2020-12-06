""" CS 254 Final Project - Classification Analysis """

### Imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model_setup
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, auc, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier  
import warnings
warnings.filterwarnings("ignore")

### Main ###
# Setting up model
nypd = model_setup.set_up()
print(len(nypd))

# Dropping Staten Island
nypd = nypd[nypd.BORO_NM != 'STATEN ISLAND']
print(len(nypd))

# Splitting and standardizing data
X_train, X_test, y_train, y_test = model_setup.split_data(nypd)
colnames = X_train.columns
X_train, X_test = model_setup.standardize(X_train, X_test)

X = pd.DataFrame(X_train)


# Getting model - makes this easy to switch out
model = model_setup.gradient_boosting()
model_name = 'GBC' # good for plotting

# Fitting model
model.fit(X_train, y_train)
# Predicting
pred = model.predict(X_test)
# Metrics and visualizations
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
plot_confusion_matrix(model, X_test, y_test)
plt.show()

# Fitting OneVsRest model
ovr = OneVsRestClassifier(estimator=model)
ovr.fit(X_train, y_train)
# Predicting
pred = ovr.predict(X_test)
# Metrics and visualizations
print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
plot_confusion_matrix(ovr, X_test, y_test)
plt.show()

# Best parameters from OvR (needs to be commented out for log_reg)
for i in range(len(ovr.estimators_)):
    print(ovr.estimators_[i].best_params_)
    
# Looking at feature importances for the ovr models
for i in range(len(ovr.estimators_)):
    feature_importances = pd.DataFrame(ovr.estimators_[i].best_estimator_.feature_importances_, index=colnames,
                                          columns=["importance"]).sort_values('importance', ascending = False)
    figure_name = 'figures/' + model_name + '_feature_importance_' + ovr.classes_[i] + '.png'
    print(model_name + ' Feature Importances for ' + ovr.classes_[i])
    print(feature_importances.head(10))
    plt.barh(feature_importances.head(10).index, feature_importances.head(10)['importance'])
    plt.gca().invert_yaxis()
    plt.grid(True, which='major', axis='both')
    plt.title(model_name + ' Feature Importance for ' + ovr.classes_[i])
    plt.xlabel('importance')
    #plt.savefig(figure_name)
    plt.show()
