# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 13:16:00 2023

@author: lb945465
"""

import pandas as pd
import os

# Define the paths to the Excel files
meteorology_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet\Merged_Meteorology.xlsx'
pollutants_path = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet\Merged_Pollutants.xlsx'
output_file = r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\State University of New York\Extra\Extra work\Mesonet\Combined_Data.xlsx'

# Get the list of sites (worksheet names) from both files
meteo_xls = pd.ExcelFile(meteorology_path)
pollutant_xls = pd.ExcelFile(pollutants_path)

meteo_sites = set(meteo_xls.sheet_names)
pollutant_sites = set(pollutant_xls.sheet_names)

# Find common sites in both files
common_sites = meteo_sites.intersection(pollutant_sites)

# Initialize an ExcelWriter for the output file
with pd.ExcelWriter(output_file) as writer:
    # Process each common site
    for site in common_sites:
        # Read the corresponding site data from both files
        meteo_df = pd.read_excel(meteorology_path, sheet_name=site, index_col='Datetime')
        pollutant_df = pd.read_excel(pollutants_path, sheet_name=site, index_col='Datetime')

        # Select only the required columns from the meteorology DataFrame
        meteo_df = meteo_df[['wdir_sonic', 'wspd_sonic']]

        # Merge the data (ensure that the index or columns to merge on are aligned)
        combined_df = meteo_df.merge(pollutant_df, left_index=True, right_index=True, how='outer')

        # Rename the index of the combined DataFrame
        combined_df.index.name = 'Datetime'

        # Write the combined DataFrame to the new Excel file
        combined_df.to_excel(writer, sheet_name=site)
