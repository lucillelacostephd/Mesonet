# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:01:51 2023
This script is to manage the mesonet datasets by concatenating all raw files and converting them to hourly data.
@author: lb945465
"""

import pandas as pd
import os
import glob

# Define the path to the folder and output Excel file
folder_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet\Data'
output_file = os.path.join(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet', 'Merged_Meteorology.xlsx')

# Get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Create an ExcelWriter object
with pd.ExcelWriter(output_file) as writer:
    # Process each file
    for file in csv_files:
        # Extract the site name from the filename
        site_name = file.split('_')[-1].split('.')[0]
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)

        # Convert the 'datetime' column to DateTime format
        df['Datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

        # Set 'datetime' as index
        df.set_index('Datetime', inplace=True)

        # Resample the data to hourly
        df = df.resample('H').mean()

        # Write the DataFrame to a worksheet named after the site_name
        df.to_excel(writer, sheet_name=site_name)
