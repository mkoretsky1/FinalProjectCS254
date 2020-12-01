""" CS 254 - Final Project - Data Cleaning/Exploration """
#import tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import model_setup

#load in the data
nypd = model_setup.set_up()
print(len(nypd))

# Getting rid of Staten Island
nypd_no_stat = nypd[nypd.BORO_NM != 'STATEN ISLAND']

#We can see here the count of non null values for each feature
print(nypd.info())

#here we can see the name of the columns
print(nypd.columns)

#here we get the value counts of the suspects race and see a majority of black
N = 6

bronx_dist =nypd[nypd.BORO_NM == "BRONX"]['VIC_RACE'].value_counts().sort_index()

queens_dist = nypd[nypd.BORO_NM == "QUEENS"]['VIC_RACE'].value_counts().sort_index()

brooklyn_dist = nypd[nypd.BORO_NM == "BROOKLYN"]['VIC_RACE'].value_counts().sort_index()

manhattan_dist = nypd[nypd.BORO_NM == "MANHATTAN"]['VIC_RACE'].value_counts().sort_index()

print(brooklyn_dist)
print(queens_dist)
print(bronx_dist)
print(manhattan_dist)

plt.figure(figsize=(10,10))
ind = np.arange(N)
width = 0.15
plt.bar(ind, bronx_dist, width, label='bronx')
plt.bar(ind + width, queens_dist, width,
    label='queens')
plt.bar(ind + width + width, brooklyn_dist, width,
    label='brooklyn')
plt.bar(ind + width + width + width, manhattan_dist, width,
    label='manhattan')

plt.ylabel('Count')
plt.title('Suspect Race by Borough')

plt.xticks(ind + width / 2, ('UNKNOWN', 'ASIAN / PACIFIC ISLANDER', 'BLACK', 'BLACK HISPANIC', 'WHITE', 'WHITE HISPANIC'))
plt.legend(loc='best')
plt.show()

# if race == 'AMERICAN INDIAN/ALAKAN NATIVE':
#     return 1
# elif race == 'ASIAN / PACIFIC ISLANDER' or race == 'ASIAN/PACIFIC ISLANDER':
#     return 2
# elif race == 'BLACK':
#     return 3
# elif race == 'BLACK HISPANIC':
#     return 4
# elif race == 'UNKNOWN':
#     return 0
# elif race == 'WHITE':
#     return 5
# elif race == 'WHITE HISPANIC':
#     return 6

#might need to make a key for the three digit offense clarrification code
#print(df['KY_CD'].value_counts())
#would be hard to make a key but this could a good feature

#need to check if PD_DESC is too variable
#print(df['PD_DESC'].value_counts())
#231 unique values so not bad could be used but would create 231 new features which is alot


#level of offense(LAW_CAT_CD) will probably be useful. maybe more useful than crime description
#print(df['LAW_CAT_CD'].value_counts()) #will make one hot encoding of this




