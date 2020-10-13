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
