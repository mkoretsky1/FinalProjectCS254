""" CS 254 - Final Project - Data Cleaning/Exploration """
#import tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_setup
from sklearn.decomposition import PCA

#load in the data
nypd = model_setup.set_up()
print(len(nypd))

# Getting rid of Staten Island
nypd_no_stat = nypd[nypd.BORO_NM != 'STATEN ISLAND']

# Splitting data
X_train, X_test, y_train, y_test = model_setup.split_data(nypd)

# Standardizing data
X_train, X_test = model_setup.standardize(X_train, X_test)

#We can see here the count of non null values for each feature
print(nypd.info())

#here we can see the name of the columns
print(nypd.columns)

#lets try looking at the split of the boros with PCA
#transform the data to two components
clf = PCA(n_components=2)
transformed_data = clf.fit_transform(X_train)

# #create a scatter plot with different colors for different clases of data-points
# class_0 = np.where(y_train == "BRONX")
# class_1 = np.where(y_train == "BROOKLYN")
# class_2 = np.where(y_train == "QUEENS")
# class_3 = np.where(y_train == "STATEN ISLAND")
# class_4 = np.where(y_train == "MANHATTAN")
#
# # plotting of transformed data by class
# fig, ax = plt.subplots(figsize=(10, 8))
# plt.scatter(transformed_data[:, 0][class_0], transformed_data[:, 1][class_0], label = "Bronx")
# plt.scatter(transformed_data[:, 0][class_1], transformed_data[:, 1][class_1], label = "Brooklyn")
# plt.scatter(transformed_data[:, 0][class_2], transformed_data[:, 1][class_2], label = "Queens" )
# plt.scatter(transformed_data[:, 0][class_3], transformed_data[:, 1][class_3], label = "Staten Island ")
# plt.scatter(transformed_data[:, 0][class_4], transformed_data[:, 1][class_4], label = "Manhatten")
# plt.legend()
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.show()

# #bar plot of the distributions of races by borough
# N = 6
#
# bronx_dist =nypd[nypd.BORO_NM == "BRONX"]['VIC_RACE'].value_counts().sort_index()
#
# queens_dist = nypd[nypd.BORO_NM == "QUEENS"]['VIC_RACE'].value_counts().sort_index()
#
# brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]['VIC_RACE'].value_counts().sort_index()
#
# manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]['VIC_RACE'].value_counts().sort_index()
#
# print(brooklyn_dist)
# print(queens_dist)
# print(bronx_dist)
# print(manhattan_dist)
#
# plt.figure(figsize=(10,10))
# ind = np.arange(N)
# width = 0.15
# plt.bar(ind, bronx_dist, width, label='bronx')
# plt.bar(ind + width, queens_dist, width,
#     label='queens')
# plt.bar(ind + width + width, brooklyn_dist, width,
#     label='brooklyn')
# plt.bar(ind + width + width + width, manhattan_dist, width,
#     label='manhattan')
#
# plt.ylabel('Count')
# plt.title('Suspect Race by Borough')
#
# plt.xticks(ind + width / 2, ('UNKNOWN', 'ASIAN / PACIFIC ISLANDER', 'BLACK', 'BLACK HISPANIC', 'WHITE', 'WHITE HISPANIC'))
# plt.legend(loc='best')
# plt.show()
#
# #bar plot of the distribution of boroughs
#
#
# x = ['Brooklyn', 'Manhattan', 'Bronx', 'Queens', 'Staten Island']
# boro_dist =nypd['BORO_NM'].value_counts()
#
# x_pos = [i for i, _ in enumerate(x)]
#
# plt.bar(x_pos, boro_dist, color='green')
# plt.xlabel("New York City Borough")
# plt.ylabel("Crime Count")
# plt.title("Distribution of Crime Across NYC Boroughs")
#
# plt.xticks(x_pos, x)
#
# plt.show()
#
# #IMPORTANT FEATURES FROM FEATURE IMPORTANCE
#
# #BRONX: off_granular_WEAPONS, POSSESSION, ETC
#
# #BROOKLYN: off_granular_ROBBERY,OPEN AREA UNCLASSIFIED
#
# #MANHATTAN: off_granular_LARCENY,GRAND FROM BUILDING (NON-RESIDENCE) UNATTENDED, off_GRAND LARCENY
#
# #QUEENS: off_granular_MARIJUANA, POSSESSION 4 & 5, off_granular_LARCENY,GRAND OF AUTO
#
# #bar plot of the distributions of important crime features by borough
#
#BRONX
bronx_dist =nypd[nypd.BORO_NM == "BRONX"]["off_DANGEROUS DRUGS"].sum()

queens_dist = nypd[nypd.BORO_NM == "QUEENS"]["off_DANGEROUS DRUGS"].sum()

brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]["off_DANGEROUS DRUGS"].sum()

manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]["off_DANGEROUS DRUGS"].sum()

print(brooklyn_dist)
print(queens_dist)
print(bronx_dist)
print(manhattan_dist)
x = ['Bronx', 'Queens', 'Brooklyn', 'Manhattan']
x_pos = [i for i, _ in enumerate(x)]
nyc_dist = [bronx_dist, queens_dist, brooklyn_dist, manhattan_dist]

plt.bar(x_pos, nyc_dist, color='green')
plt.xticks(x_pos, x)
plt.ylabel('Crime Count')
plt.xlabel("New York City Borough")
plt.title('Bronx Important Feature: Weapon Possession Calls')

plt.show()
#
# #BROOKLYN
#
# bronx_dist =nypd[nypd.BORO_NM == "BRONX"]["off_granular_ROBBERY,OPEN AREA UNCLASSIFIED"].sum()
#
# queens_dist = nypd[nypd.BORO_NM == "QUEENS"]["off_granular_ROBBERY,OPEN AREA UNCLASSIFIED"].sum()
#
# brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]["off_granular_ROBBERY,OPEN AREA UNCLASSIFIED"].sum()
#
# manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]["off_granular_ROBBERY,OPEN AREA UNCLASSIFIED"].sum()
#
# print(brooklyn_dist)
# print(queens_dist)
# print(bronx_dist)
# print(manhattan_dist)
# x = ['Bronx', 'Queens', 'Brooklyn', 'Manhattan']
# x_pos = [i for i, _ in enumerate(x)]
# nyc_dist = [bronx_dist, queens_dist, brooklyn_dist, manhattan_dist]
#
# plt.bar(x_pos, nyc_dist, color='green')
# plt.xticks(x_pos, x)
# plt.ylabel('Crime Count')
# plt.xlabel("New York City Borough")
# plt.title('Brooklyn Important Feature: Robbery in Open Area')
#
# plt.show()
#
# #MANHATTAN
#
# bronx_dist =nypd[nypd.BORO_NM == "BRONX"]["off_granular_LARCENY,GRAND FROM BUILDING (NON-RESIDENCE) UNATTENDED"].sum()
#
# queens_dist = nypd[nypd.BORO_NM == "QUEENS"]["off_granular_LARCENY,GRAND FROM BUILDING (NON-RESIDENCE) UNATTENDED"].sum()
#
# brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]["off_granular_LARCENY,GRAND FROM BUILDING (NON-RESIDENCE) UNATTENDED"].sum()
#
# manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]["off_granular_LARCENY,GRAND FROM BUILDING (NON-RESIDENCE) UNATTENDED"].sum()
#
# print(brooklyn_dist)
# print(queens_dist)
# print(bronx_dist)
# print(manhattan_dist)
# x = ['Bronx', 'Queens', 'Brooklyn', 'Manhattan']
# x_pos = [i for i, _ in enumerate(x)]
# nyc_dist = [bronx_dist, queens_dist, brooklyn_dist, manhattan_dist]
#
# plt.bar(x_pos, nyc_dist, color='green')
# plt.xticks(x_pos, x)
# plt.ylabel('Crime Count')
# plt.xlabel("New York City Borough")
# plt.title('Manhattan Important Feature: Grand Larceny From Unattended Building')
#
# plt.show()
#
# #MANHATTAN
#
# bronx_dist =nypd[nypd.BORO_NM == "BRONX"]["off_GRAND LARCENY"].sum()
#
# queens_dist = nypd[nypd.BORO_NM == "QUEENS"]["off_GRAND LARCENY"].sum()
#
# brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]["off_GRAND LARCENY"].sum()
#
# manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]["off_GRAND LARCENY"].sum()
#
# print(brooklyn_dist)
# print(queens_dist)
# print(bronx_dist)
# print(manhattan_dist)
# x = ['Bronx', 'Queens', 'Brooklyn', 'Manhattan']
# x_pos = [i for i, _ in enumerate(x)]
# nyc_dist = [bronx_dist, queens_dist, brooklyn_dist, manhattan_dist]
#
# plt.bar(x_pos, nyc_dist, color='green')
# plt.xticks(x_pos, x)
# plt.ylabel('Crime Count')
# plt.xlabel("New York City Borough")
# plt.title('Manhattan Important Feature: Grand Larceny')
#
# plt.show()
#
#
# #QUEENS
#
# bronx_dist =nypd[nypd.BORO_NM == "BRONX"]["off_granular_MARIJUANA, POSSESSION 4 & 5"].sum()
#
# queens_dist = nypd[nypd.BORO_NM == "QUEENS"]["off_granular_MARIJUANA, POSSESSION 4 & 5"].sum()
#
# brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]["off_granular_MARIJUANA, POSSESSION 4 & 5"].sum()
#
# manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]["off_granular_MARIJUANA, POSSESSION 4 & 5"].sum()
#
# print(brooklyn_dist)
# print(queens_dist)
# print(bronx_dist)
# print(manhattan_dist)
# x = ['Bronx', 'Queens', 'Brooklyn', 'Manhattan']
# x_pos = [i for i, _ in enumerate(x)]
# nyc_dist = [bronx_dist, queens_dist, brooklyn_dist, manhattan_dist]
#
# plt.bar(x_pos, nyc_dist, color='green')
# plt.xticks(x_pos, x)
# plt.ylabel('Crime Count')
# plt.xlabel("New York City Borough")
# plt.title('Queens Important Feature: Marijuana Possession')
#
# plt.show()
#






