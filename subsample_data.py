""" CS 254 - Final Project - Data Cleaning/Exploration """

### Imports ###
import pandas as pd

df = pd.read_csv('nypd_data/NYPD_Complaint_Data_Historic.csv')

df_10000 = df.sample(n=10000)
df_100000 = df.sample(n=100000)
df_1000000 = df.sample(n=1000000)

df_10000.to_csv('nypd_data/NYPD_Complaint_Data_Historic_10000_subsamples.csv')
df_100000.to_csv('nypd_data/NYPD_Complaint_Data_Historic_100000_subsamples.csv')
df_1000000.to_csv('nypd_data/NYPD_Complaint_Data_Historic_1000000_subsamples.csv')

df = pd.read_csv('nypd_data/NYPD_Complaint_Data_Historic_10000_subsamples.csv')
