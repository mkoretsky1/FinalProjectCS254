""" CS 254 - Final Project - Data Cleaning/Exploration """
#import tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#read in subsample
df = pd.read_csv('nypd_data/NYPD_Complaint_Data_Historic_10000_subsamples.csv')

#We can see here the count of non null values for each feature
print(df.info())

#here we can see the name of the columns
print(df.columns)

#here we get the value counts of the suspects race and see a majority of black
print(df['SUSP_RACE'].value_counts())

#From this we can see that there are man columns that we can get rid since there there are many null values

#complaint start date and time is important

#complaint end date and time might not be as important but could create feature
#that is length of the complaint

#Addr_PCT_CD is the precinct which it occured and might be dropped if there is a bureau location(DROP)
df = df.drop("ADDR_PCT_CD", axis=1)

#rpt_date is redundant and may be dropped
df = df.drop("RPT_DT", axis=1)

#might need to make a key for the three digit offense clarrification code
print(df['KY_CD'].value_counts())
#would be hard to make a key but this could a good feature

#need to check if PD_DESC is too variable
print(df['PD_DESC'].value_counts())
df = df.drop('PD_DESC', axis=1)
#231 unique values so not bad could be used but would create 231 new features which is alot

#CRM_ATPT_CPTD_CD is indicator of whether crime was completed or not. id say
#its useful

#level of offense(LAW_CAT_CD) will probably be useful. maybe more useful than crime description
print(df['LAW_CAT_CD'].value_counts()) #will make one hot encoding of this

#boro_nm is much more important than the other locations in this data set

#need to check jurisdiction code and description to see if useful
print(df['JURIS_DESC'].value_counts())
print(df['JURISDICTION_CODE'].value_counts())
#I dont think we care about jurisdiction
df = df.drop('JURIS_DESC', axis=1)
df = df.drop('JURISDICTION_CODE', axis=1)

#parks_nm is not useful
df = df.drop('PARKS_NM', axis=1)

#HADEVElOPT is not useful
df = df.drop('HADEVELOPT', axis=1)

#housing development level code may be important but lots of null
df = df.drop('HOUSING_PSA', axis=1)

#X_cord Y_cord of city will not be important
df = df.drop('X_COORD_CD', axis=1)
df = df.drop('Y_COORD_CD', axis=1)

#Suspect age group is important

#Suspect sex could be important

#latitude longitude will be important for mapping

#drop partrol_boro
df = df.drop('PATROL_BORO', axis=1)

#drop station name
df = df.drop('STATION_NAME', axis=1)

#keep vic age group

#keep vic race

#keep vic sex

#taken from 35 to 23 features
print(df.info())



