""" CS 254 - Final Project - Data Cleaning/Exploration """
#import tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

### Functions ###
def age_groups(age):
    if age == '<18' or age =='18-24' or age == '25-44' or age == '45-64' or age == '65+':
        return age
    else:
        return 'UNKNOWN'

#read in subsample
df = pd.read_csv('nypd_data/NYPD_Complaint_Data_Historic_10000_subsamples.csv')

#We can see here the count of non null values for each feature
print(df.info())

#here we can see the name of the columns
print(df.columns)

#here we get the value counts of the suspects race and see a majority of black
print(df['SUSP_RACE'].value_counts())

#From this we can see that there are many columns that we can get rid since there there are many null values

#complaint start date and time is important

#complaint end date and time might not be as important but could create feature
#that is length of the complaint

#Addr_PCT_CD is the precinct which it occured and might be dropped if there is a bureau location(DROP)

#rpt_date is redundant and may be dropped

#might need to make a key for the three digit offense clarrification code
print(df['KY_CD'].value_counts())
#would be hard to make a key but this could a good feature

#need to check if PD_DESC is too variable
print(df['PD_DESC'].value_counts())
#231 unique values so not bad could be used but would create 231 new features which is alot

#CRM_ATPT_CPTD_CD is indicator of whether crime was completed or not. id say
#its useful

#level of offense(LAW_CAT_CD) will probably be useful. maybe more useful than crime description
print(df['LAW_CAT_CD'].value_counts()) #will make one hot encoding of this

#boro_nm is much more important than the other locations in this data set

#need to check jurisdiction code and description to see if useful

#I dont think we care about jurisdiction

#parks_nm is not useful

#HADEVElOPT is not useful

#housing development level code may be important but lots of null

#X_cord Y_cord of city will not be important

#Suspect age group is important

#Suspect sex could be important

#latitude longitude will be important for mapping

#drop partrol_boro

#drop station name

#keep vic age group

#keep vic race

#keep vic sex

#Drop the unecessary features
drop = ['Unnamed: 0', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'PARKS_NM', 'HADEVELOPT', 'HOUSING_PSA',
         'TRANSIT_DISTRICT', 'STATION_NAME', 'JURIS_DESC', 'JURISDICTION_CODE', 'RPT_DT', 'PATROL_BORO',
         'X_COORD_CD', 'Y_COORD_CD', 'Lat_Lon', 'PD_CD', 'PD_DESC', 'KY_CD','ADDR_PCT_CD']


nypd = df.drop(drop, axis=1)

# Extracting each piece of datetime
nypd['CMPLNT_FR_DT'] = nypd.CMPLNT_FR_DT.replace({'1019':'2019', '1016':'2016', '1017':'2017'}, regex = True)
nypd['year'] = pd.DatetimeIndex(nypd['CMPLNT_FR_DT']).year
nypd['month'] = pd.DatetimeIndex(nypd['CMPLNT_FR_DT']).month
nypd['day'] = pd.DatetimeIndex(nypd['CMPLNT_FR_DT']).day
nypd['hour'] = pd.DatetimeIndex(nypd['CMPLNT_FR_TM']).hour
nypd['minute'] = pd.DatetimeIndex(nypd['CMPLNT_FR_TM']).minute
nypd['second'] = pd.DatetimeIndex(nypd['CMPLNT_FR_TM']).second
# Creating new datetime object
nypd['complaint_datetime'] = pd.to_datetime(nypd[['year','month','day','hour','minute','second']])
# Mapping attepted/completed to 0/1
nypd['attempted_completed'] = nypd['CRM_ATPT_CPTD_CD'].map({'COMPLETED':1,'ATTEMPTED':0})
# Dropping old datetime columns and attemped/completed
nypd = nypd.drop(['CMPLNT_FR_DT','CMPLNT_FR_TM','CRM_ATPT_CPTD_CD'], axis=1)

# Fixing issues with age groups
nypd['SUSP_AGE_GROUP'] = nypd['SUSP_AGE_GROUP'].apply(age_groups)
nypd['VIC_AGE_GROUP'] = nypd['VIC_AGE_GROUP'].apply(age_groups)

# List of variables that need one-hot encoding
one_hot = ['OFNS_DESC','LAW_CAT_CD','BORO_NM','LOC_OF_OCCUR_DESC','PREM_TYP_DESC','SUSP_AGE_GROUP','SUSP_RACE',
           'SUSP_SEX','VIC_AGE_GROUP','VIC_RACE','VIC_SEX']

# Creating dummy variables where applicable - ignoring nan for now (can make a column for them if we want)
nypd = pd.get_dummies(nypd, columns=one_hot,
                      prefix=['off','law_cat','boro','loc','loc_type','susp_age','susp_race',
                              'susp_sex','vic_age','vic_race','vic_sex'])
# Output to csv
nypd.to_csv('nypd_data/nypd_10000.csv')