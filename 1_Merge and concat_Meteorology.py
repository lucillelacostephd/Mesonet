# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:01:51 2023
This script is to manage the mesonet datasets by concatenating all raw files and converting them to hourly data.
@author: lb945465
"""

import pandas as pd
import os
import glob

# Define the path to the folder
folder_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\NYSERDA VOC project\Data\Mesonet'

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Initialize an empty DataFrame to hold all merged data
merged_df = pd.DataFrame()

# Process each file
for file in csv_files:
    # Extract the site name from the filename
    site_name = file.split('_')[-1].split('.')[0]
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Convert the 'datetime' column to DateTime format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%dT%H%M')

    # Set 'datetime' as index
    df.set_index('datetime', inplace=True)

    # Resample the data to hourly
    df = df.resample('H').mean()
    
    # Add the 'Site' column
    df['Site'] = site_name
    
    # Append this DataFrame to the merged DataFrame
    merged_df = pd.concat([merged_df, df], ignore_index=False)

# Display the first few rows of the merged DataFrame
print(merged_df.head())

# Save the merged DataFrame to a new CSV file in the same folder
output_file = os.path.join(folder_path, 'Merged_File.csv')
merged_df.to_csv(output_file)

