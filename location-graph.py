#import tools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import folium
import folium.plugins

nypd = pd.read_csv('nypd_data/nypd_100000.csv', parse_dates=['complaint_datetime'])
# Getting X data
# Variables to drop regardless of the analysis
drop_always = ['CMPLNT_NUM','SUSP_RACE','SUSP_SEX','VIC_SEX','complaint_datetime','Unnamed: 0','Unnamed: 0.1']
# Creating one list of variables to drop - Edit this line based on analysis being performed
drop = drop_always
nypd = nypd.drop(drop, axis=1)
nypd = nypd.dropna(axis=0)

output_file = "crime_map_nyc.html"
NYC_COORDINATES = (40.7128, -74.0060)
nyc_map = folium.Map(location=NYC_COORDINATES, zoom_start=13)
locs = nypd[['Latitude', 'Longitude']].astype('float').dropna().values
heatmap = folium.plugins.HeatMap(locs.tolist(), radius = 10)
nyc_map.add_child(heatmap)

nyc_map.save(output_file)
