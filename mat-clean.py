""" CS 254 - Final Project - Data Cleaning """

### Imports ###
import pandas as pd
import numpy as np

### Functions ###
def age_groups(age):
    if age == '<18' or age =='18-24' or age == '25-44' or age == '45-64' or age == '65+':
        return age
    else:
        return 'UNKNOWN'
    
def map_race(race):
    if race == 'AMERICAN INDIAN/ALAKAN NATIVE':
        return 1
    elif race == 'ASIAN / PACIFIC ISLANDER' or race == 'ASIAN/PACIFIC ISLANDER':
        return 2
    elif race == 'BLACK':
        return 3
    elif race == 'BLACK HISPANIC':
        return 4
    elif race == 'UNKNOWN':
        return 0
    elif race == 'WHITE':
        return 5
    elif race == 'WHITE HISPANIC':
        return 6
    
def tod_groups(hour):
    if hour < 12:
        return 'morning'
    elif 12 <= hour <= 19:
        return 'afternoon'
    else:
        return'night'
        
def season_groups(month):
    if month == 12 or month <= 2:
        return 'winter'
    elif 2 < month <= 5:
        return 'spring'
    elif 5 <  month <= 8:
        return 'summer'
    else:
        return 'fall'
    
### Main ###
nypd = pd.read_csv('nypd_data/NYPD_Complaint_Data_Historic_10000_subsamples.csv', 
                         parse_dates=['CMPLNT_FR_DT','CMPLNT_FR_TM'])

# Columns to drop - mostly based on information for website where data was downloaded
# X coord and Y coord not needed when we have latitude and longitude
# PD_CD and PD_DESC too granular to make one-hot variables with
drop = ['CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'PARKS_NM', 'HADEVELOPT', 'HOUSING_PSA', 
         'TRANSIT_DISTRICT', 'STATION_NAME', 'JURIS_DESC', 'JURISDICTION_CODE', 'RPT_DT', 'PATROL_BORO',
         'X_COORD_CD', 'Y_COORD_CD', 'Lat_Lon', 'PD_CD', 'KY_CD','ADDR_PCT_CD']
nypd = nypd.drop(drop, axis=1)

# Extracting each piece of datetime
nypd['CMPLNT_FR_DT'] = nypd.CMPLNT_FR_DT.replace({'1019':'2019', '1016':'2016', '1017':'2017',
                                                  '1015':'2015','1018':'2018','1028':'2018',
                                                  '1027':'2017','1026':'2016','1025':'2015',
                                                  '1029':'2019'}, regex = True)
nypd['year'] = pd.DatetimeIndex(nypd['CMPLNT_FR_DT']).year
nypd['month'] = pd.DatetimeIndex(nypd['CMPLNT_FR_DT']).month
nypd['day'] = pd.DatetimeIndex(nypd['CMPLNT_FR_DT']).day
nypd['hour'] = pd.DatetimeIndex(nypd['CMPLNT_FR_TM']).hour
nypd['minute'] = pd.DatetimeIndex(nypd['CMPLNT_FR_TM']).minute
# Creating new datetime object
nypd['complaint_datetime'] = pd.to_datetime(nypd[['year','month','day','hour','minute']])
# Mapping attepted/completed to 0/1
nypd['attempted_completed'] = nypd['CRM_ATPT_CPTD_CD'].map({'COMPLETED':1,'ATTEMPTED':0})
# Mapping violation/misdemeanor/felony to 0/1/2
nypd['LAW_CAT_CD'] = nypd['LAW_CAT_CD'].map({'VIOLATION':0,'MISDEMEANOR':1,'FELONY':2})

# Dropping old datetime columns and attemped/completed
nypd = nypd.drop(['CMPLNT_FR_DT','CMPLNT_FR_TM','CRM_ATPT_CPTD_CD'], axis=1)

# Fixing issues with age groups
nypd['SUSP_AGE_GROUP'] = nypd['SUSP_AGE_GROUP'].apply(age_groups)
nypd['VIC_AGE_GROUP'] = nypd['VIC_AGE_GROUP'].apply(age_groups)

# Mapping age variables
nypd['SUSP_AGE_GROUP'] = nypd['SUSP_AGE_GROUP'].map({'UNKNOWN':0,'<18':1,'18-24':2,'25-44':3,'45-64':4,'65+':5})
nypd['VIC_AGE_GROUP'] = nypd['VIC_AGE_GROUP'].map({'UNKNOWN':0,'<18':1,'18-24':2,'25-44':3,'45-64':4,'65+':5})

# Mapping sex variables
nypd['SUSP_SEX'] = nypd['SUSP_SEX'].map({'UNKNOWN':0,'D':1,'E':2,'F':3,'M':4})
nypd['VIC_SEX'] = nypd['SUSP_SEX'].map({'UNKNOWN':0,'D':1,'E':2,'F':3,'M':4})

# Mapping race variables
nypd['SUSP_RACE'] = nypd['SUSP_RACE'].apply(map_race)
nypd['VIC_RACE'] = nypd['VIC_RACE'].apply(map_race)

# Mapping general location variables
nypd['LOC_OF_OCCUR_DESC'] = nypd['LOC_OF_OCCUR_DESC'].map({'FRONT OF':0,'INSIDE':1,'OPPOSITE OF':2,
                                                           'OUTSIDE':3,'REAR OF':4})

# Creating categorical variables for time of day and season
# nypd['time_of_day'] = nypd['hour'].apply(tod_groups)
# nypd['season'] = nypd['month'].apply(season_groups)

# Storing borough name to add back after dummy variable creation
borough_name = nypd['BORO_NM'].values

# List of variables that need one-hot encoding - might want to add year, month to this
one_hot = ['OFNS_DESC','PD_DESC','BORO_NM','PREM_TYP_DESC']
# Uncomment to check the unique values of each one-hot variable
# for var in one_hot:
#     print(var)
#     print(nypd[var].unique())
    
# Creating dummy variables where applicable - ignoring nan for now (can make a column for them if we want)
nypd = pd.get_dummies(nypd, columns=one_hot, 
                      prefix=['off','off_granular','boro','loc_type'])

# Adding borough name back in
nypd['BORO_NM'] = borough_name

#deleting the added unnamed feature
print(nypd.head())

nypd = nypd.drop(["Unnamed: 0"], axis=1)

print(nypd.head())

# Output to csv
nypd.to_csv('nypd_data/nypd_10000.csv')





