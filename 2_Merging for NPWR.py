# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:12:25 2023
This script is to merge wind speed and direction dataset with the PMF source
contribution dataset. Use absolute concentrations for source contributions, 
this is the Contributions_conc worksheet of the Base results excel file.
@author: lb945465
"""

import pandas as pd

# Load the dataframe with columns ws_ms and wd
df1 = pd.read_csv(r'C:\Users\LB945465\OneDrive - University at Albany - SUNY\NYSERDA VOC project\LBORLAZA\Mesonet\Data\Merged_File.csv', usecols=["datetime", "wspd_sonic", "wdir_sonic"])  
df1=df1.set_index("datetime")
df1=df1.rename_axis("Datetime")

# Load the excel worksheet as a dataframe
excel_file = pd.ExcelFile(r"C:\Users\LB945465\OneDrive - University at Albany - SUNY\NYSERDA VOC project\LBORLAZA\Mesonet\Data\Pollutants\Merged_Pollutants.xlsx")  # Change the file name and path accordingly
df2 = excel_file.parse("pm25")
df2=df2.set_index("Datetime")

# Merge the dataframes using the shared index "Date"
merged_df = df1.merge(df2, left_index=True, right_index=True)

# Clean the merged_df by dropping rows with ws_ms less than 1 or equal to 0
merged_df = merged_df[(merged_df["wspd_sonic"] > 0) & (merged_df["wspd_sonic"] >= 1)]

# Clean the merged_df by dropping rows with NaN in ws_ms or wd
merged_df = merged_df.dropna(subset=["wspd_sonic", "wdir_sonic"])

# Rename specific columns in merged_df
merged_df.rename(columns={'wspd_sonic': 'ws', 
                               }, inplace=True)

# Display the merged dataframe
print(merged_df)

# Save the cleaned and merged dataframe to an Excel file
output_file = "NPWR_merged.csv"
merged_df.to_csv(output_file, index=True)

print("Cleaned and merged dataframe saved to", output_file)

