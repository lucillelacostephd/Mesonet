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

# Get all Excel files in the folder
excel_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Create a Pandas Excel writer using XlsxWriter as the engine
output_file = os.path.join(folder_path, 'Merged_Pollutants.xlsx')
writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

# Process each file
for file in excel_files:
    # Extract the variable name from the filename (e.g., 'CO' from 'CO.csv')
    variable_name = os.path.basename(file).split('.')[0]

    # Read the Excel file into a DataFrame
    df = pd.read_csv(file)

    # Add the 'variable' column
    df['variable'] = variable_name

    # Convert the 'Datetime' column to DateTime format and set as index
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    # Write each DataFrame to a different worksheet
    df.to_excel(writer, sheet_name=variable_name, index=True)

# Close the Pandas Excel writer and save the Excel file
writer.save()
