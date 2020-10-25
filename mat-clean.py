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
    
### Main ###
nypd = pd.read_csv('nypd_data/NYPD_Complaint_Data_Historic_10000_subsamples.csv', 
                         parse_dates=['CMPLNT_FR_DT','CMPLNT_FR_TM'])

# Columns to drop - mostly based on information for website where data was downloaded
# X coord and Y coord not needed when we have latitude and longitude
# PD_CD and PD_DESC too granular to make one-hot variables with
drop = ['Unnamed: 0', 'CMPLNT_TO_DT', 'CMPLNT_TO_TM', 'PARKS_NM', 'HADEVELOPT', 'HOUSING_PSA', 
         'TRANSIT_DISTRICT', 'STATION_NAME', 'JURIS_DESC', 'JURISDICTION_CODE', 'RPT_DT', 'PATROL_BORO',
         'X_COORD_CD', 'Y_COORD_CD', 'Lat_Lon', 'PD_CD', 'PD_DESC', 'KY_CD','ADDR_PCT_CD']
nypd = nypd.drop(drop, axis=1)

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
# Uncomment to check the unique values of each one-hot variable
# for var in one_hot:
#     print(var)
#     print(nypd[var].unique())
    
# Creating dummy variables where applicable - ignoring nan for now (can make a column for them if we want)
nypd = pd.get_dummies(nypd, columns=one_hot, 
                      prefix=['off','law_cat','boro','loc','loc_type','susp_age','susp_race',
                              'susp_sex','vic_age','vic_race','vic_sex'])
# Output to csv
nypd.to_csv('nypd_data/nypd_10000.csv')





