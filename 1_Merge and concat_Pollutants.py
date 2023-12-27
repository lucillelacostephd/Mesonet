# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:58:49 2023

@author: lb945465
"""

import pandas as pd
import os
import glob

# Define the path to the folder with Excel files
folder_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet\Data\Pollutants'

output_file = os.path.join(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet', 'Merged_Pollutants.xlsx')

# Get all CSV files in the folder
excel_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Initialize a dictionary to hold data for each site
site_data = {}

# Process each pollutant file
for file in excel_files:
    # Extract pollutant name from the filename
    pollutant_name = os.path.basename(file).split('.')[0]

    # Read the CSV file into a DataFrame with 'Datetime' as index
    df = pd.read_csv(file, index_col='Datetime')

    # Transpose the DataFrame so that sites are rows and datetime is columns
    df = df.transpose()

    # Append or merge this DataFrame to each site's DataFrame in the dictionary
    for site in df.index:
        if site not in site_data:
            site_data[site] = pd.DataFrame()
        site_data[site][pollutant_name] = df.loc[site]

# Write each site's DataFrame to a separate worksheet in an Excel file
with pd.ExcelWriter(output_file) as writer:
    for site, data in site_data.items():
        data.to_excel(writer, sheet_name=site)
