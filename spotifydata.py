# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 20:13:00 2022

@author: John S. Strada
Description: This script is meant to open a larger spotify csv file, read it, 
& apply a condition basis and then write the partioned data to a new smaller spotify csv file.
"""
import pandas as pd #imports the panda module
data = pd.read_csv("C:\\Data\\spotify_songs.csv", encoding="utf8") # opens the original csv for reading and creates the data frame
#data_filtered = data[(data.region == 'United States') & (data.chart == 'top200')] # creates a filtered data frame based on conditions
#data_filtered.to_csv("C:\\Data\\songsupdate.csv", index=False) # writes filtered data frame to new csv file
data.to_csv("C:\\Data\\spotify_songsupdate.csv", encoding="utf8", index=False)
# data.groupby('year').sample(50000) additional code that can be used to sample the dataset