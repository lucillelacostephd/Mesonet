# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:41:44 2024
Script to consolidate ozone, NOx, and NOy data. 
@author: lb945465
"""

import pandas as pd

# Load the datasets
file_path = r'.\O3_NOy_NOx.xlsx' # Insert file path here

# Function to read specific columns from a worksheet
def read_specific_columns(sheet_name, columns, file_path):
    return pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns)

# Columns to load
columns_to_load = ['Value.parameter', 'Value.date_local', 'Value.time_local', 'Value.sample_measurement'] # Insert all columns to be loaded

def process_parameter_data(parameter_name, file_path, columns_to_load):
    # Initialize an empty dataframe
    parameter_data = pd.DataFrame()

    # Process each worksheet and append the data
    for year in range(2000, 2021):
        year_data = read_specific_columns(str(year), columns_to_load, file_path)
        parameter_data = pd.concat([parameter_data, year_data], ignore_index=True)

    # Filter the dataframe for the specified parameter
    parameter_only_data = parameter_data[parameter_data['Value.parameter'] == parameter_name]

    # Convert 'Value.date_local' and 'Value.time_local' into a single datetime column and set it as index
    parameter_only_data['Date'] = pd.to_datetime(parameter_only_data['Value.date_local'] + ' ' + parameter_only_data['Value.time_local'])
    parameter_only_data.set_index('Date', inplace=True)

    # Drop unnecessary columns
    parameter_only_data.drop(['Value.parameter', 'Value.date_local', 'Value.time_local'], axis=1, inplace=True)

    # Rename 'Value.sample_measurement' to the parameter name
    parameter_only_data.rename(columns={'Value.sample_measurement': parameter_name}, inplace=True)

    return parameter_only_data

# Apply the function for each parameter
ozone_data = process_parameter_data('Ozone', file_path, columns_to_load)
nitric_oxide_data = process_parameter_data('Nitric oxide (NO)', file_path, columns_to_load)
noy_data = process_parameter_data('Reactive oxides of nitrogen (NOy)', file_path, columns_to_load)

# Remove zeroes and negatives 
ozone_data = ozone_data[ozone_data != 0]
nitric_oxide_data = nitric_oxide_data[nitric_oxide_data != 0]
noy_data = noy_data[noy_data != 0]

# Convert to ppbv
ozone_data=ozone_data*1000
nitric_oxide_data=nitric_oxide_data*1000
noy_data=noy_data*1000

# Drop all rows with Nan
ozone_data.dropna(inplace=True)
nitric_oxide_data.dropna(inplace=True)
noy_data.dropna(inplace=True)

# Save dataframe to excel
ozone_data.to_excel("Ozone.xlsx")
nitric_oxide_data.to_excel("NOx.xlsx")
noy_data.to_excel("NOy.xlsx")