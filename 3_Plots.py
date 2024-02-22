# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:19:09 2024
Preparing the Mesonet/Micronet data. This code is to characterize hourly observation using calibrated sensor data at 37 NYSM and NYCM sites 
•	Diurnal trends for the whole time period
•	Weekend-weekday profiles
•	Monthly and seasonal profiles at neighborhood levels
•	Temporal (and spatial) trends :NOT DONE
•	Calculate COD/r2 to determine spatial variation across urban sites
@author: lb945465
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Directory where your CSV files are stored
directory_path = 'C:/Users/LB945465/OneDrive - University at Albany - SUNY/State University of New York/Extra/Extra work/Mesonet/Data/level2'

# Initialize an empty list to store the dataframes
dataframes = []

# Option to use all csv files or select specific ones
use_all_files = True  # Set to False if you want to select specific files
selected_files = ['site1.csv', 'site2.csv', 'site3.csv']  # Add file names for the files you want to include

# Loop through each file in the directory
for file in os.listdir(directory_path):
    if file.endswith(".csv"):
        # Check if the file is in the selected files list (if not using all files)
        if not use_all_files and file not in selected_files:
            continue  # Skip files that are not in the selected_files list
        
        file_path = os.path.join(directory_path, file)
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col='datetime', parse_dates=['datetime'])
        
        # Remove the file extension to get the site code
        site_code = file.rsplit('.', 1)[0]  # This removes the .csv extension
        
        # Add the site code as a new column in the dataframe
        df['Site'] = site_code
        
        # Append the dataframe to the list
        dataframes.append(df)

# Concatenate all dataframes into a single dataframe
consolidated_df = pd.concat(dataframes, axis=0)

# If any of the specified columns are missing in some files, this ensures they are included as NaNs
consolidated_df = consolidated_df.reindex(columns=['O3', 'CO', 'pm25', 'Site'])
consolidated_df.rename(columns={'pm25': 'PM2.5'}, inplace=True)

# List of columns to convert to numeric
columns_to_convert = ['O3', 'CO', 'PM2.5']

# Convert specified columns to numeric, coercing errors to NaN
for column in columns_to_convert:
    consolidated_df[column] = pd.to_numeric(consolidated_df[column], errors='coerce')

# Now perform the division by 1000 to convert from ppb to ppm
consolidated_df['CO'] = consolidated_df['CO'] / 1000

print(consolidated_df.head())  # Print the first few rows of the consolidated dataframe

# # Optionally, save the consolidated dataframe to a new CSV file
# consolidated_df.to_csv('C:/Users/LB945465/OneDrive - University at Albany - SUNY/State University of New York/Extra/Extra work/Mesonet/Data/level2/consolidated_data.csv')

# Reset index to make 'datetime' a column again for melting
df_reset = consolidated_df.reset_index()

# Melt the DataFrame
df_melted = pd.melt(df_reset, id_vars=['datetime', 'Site'], value_vars=['O3', 'CO', 'PM2.5'], var_name='Pollutant', value_name='Value')

# Extract hour from datetime
df_melted['Hour'] = df_melted['datetime'].dt.hour

# Add a new column for the day of the week
df_melted['Day'] = df_melted['datetime'].dt.day_name()

# Define a function or use lambda to categorize days into 'Weekday' or 'Weekend'
df_melted['Day_Category'] = df_melted['Day'].apply(lambda x: 'Weekend' if x in ['Saturday', 'Sunday'] else 'Weekday')

# Add a column for month
df_melted['month'] = df_melted['datetime'].dt.month

# Add a column for month names
df_melted['Month'] = df_melted['datetime'].dt.strftime('%b')

# Add a column for season
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df_melted['Season'] = df_melted['month'].apply(get_season)

# Drop rows with NaN in 'Value' if any resulted from conversion
df_melted.dropna(subset=['Value'], inplace=True)

# Then I'll drop all rows that have a negative 'Value'
df_melted = df_melted[df_melted['Value'] >= 0]

# Print the updated DataFrame to verify
print(df_melted.head())

# Units for each pollutant
units = {
    'O3': 'ppb',
    'CO': 'ppm',
    'PM2.5': 'μg m${^{-3}}$',
}

#########################################################################################################################################        
def plot_diurnal_trends_per_site_and_pollutant(df_melted, dpi=200):
    pollutants = df_melted['Pollutant'].unique()
    sites = df_melted['Site'].unique()
     
    # Loop through each pollutant and each site
    for pollutant in pollutants:
        for site in sites:
            df_site_pollutant = df_melted[(df_melted['Pollutant'] == pollutant) & (df_melted['Site'] == site)]
            
            plt.figure(figsize=(10, 6), dpi=dpi)
            sns.lineplot(data=df_site_pollutant, x='Hour', y='Value', marker='o', dashes=False, errorbar=None)
            
            plt.title(f'Diurnal Trends of {pollutant} at {site}')
            plt.xlabel('Hour of the Day')
            plt.ylabel(f'Concentration ({units[pollutant]})')
            plt.tight_layout()
            plt.show()

# Assuming 'df_melted' is your melted DataFrame
plot_diurnal_trends_per_site_and_pollutant(df_melted)

def plot_diurnal_trends_per_pollutant(df_melted, dpi=200):
        
    pollutants = df_melted['Pollutant'].unique()
    
    # Generate a unique color for each site using a colormap
    num_sites = len(df_melted['Site'].unique())
    colors = sns.color_palette("hsv", num_sites)  # Using the HSV color palette to generate unique colors
    
    for pollutant in pollutants:
        df_pollutant = df_melted[df_melted['Pollutant'] == pollutant]
        
        plt.figure(figsize=(15, 5), dpi=dpi)  # Set DPI to the specified value
        sns.lineplot(data=df_pollutant, x='Hour', y='Value', hue='Site', marker='o', dashes=False, errorbar=None, palette=colors)
        
        plt.title(f'Diurnal Trends of {pollutant}')
        plt.xlabel('Hour of the Day')
        plt.ylabel(f'Concentration ({units[pollutant]})')  # Dynamically set y-label with unit
        plt.legend(title='Site', bbox_to_anchor=(1, 1), loc='upper left', ncol=2)  # Adjust legend
        plt.tight_layout()
        plt.show()  # Show plot for each pollutant

# Assuming 'df_melted' is your melted DataFrame
plot_diurnal_trends_per_pollutant(df_melted)

# Redefining the function with confidence interval for all pollutants
def plot_avg_confidence_interval_for_all_pollutants(df_melted, confidence_level=0.95, dpi=200):
    # Ensure 'Value' is numeric, converting non-numeric to NaN
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')

    # Drop rows with NaN values
    df_melted.dropna(subset=['Value'], inplace=True)

    # Drop rows with negative values
    df_melted = df_melted[df_melted['Value'] >= 0]

    # List of pollutants
    pollutants = df_melted['Pollutant'].unique()
    
    for pollutant in pollutants:
        # Calculate mean and 95% CI for each hour for the current pollutant
        grouped = df_melted[df_melted['Pollutant'] == pollutant].groupby('Hour')
        
        means = grouped['Value'].mean()
        sems = grouped['Value'].sem()  # Standard error of the mean
        ci_lower, ci_upper = stats.t.interval(alpha=confidence_level, df=grouped['Value'].count()-1, loc=means, scale=sems)

        plt.figure(figsize=(12, 6), dpi=dpi)

        # Plot mean value
        plt.plot(means.index, means, label='Mean', color='blue')

        # Add shading for the 95% CI range
        plt.fill_between(means.index, ci_lower, ci_upper, color='blue', alpha=0.2)
        
        plt.title(f'Hourly Mean Concentration of {pollutant} with 95% Confidence Interval')
        plt.xlabel('Hour of the Day')
        plt.ylabel(f'Concentration ({units[pollutant]})')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Call the function with the melted DataFrame
plot_avg_confidence_interval_for_all_pollutants(df_melted)

def plot_diurnal_trends_per_site_and_pollutant_per_day(df_melted, dpi=200):
    pollutants = df_melted['Pollutant'].unique()
    sites = df_melted['Site'].unique()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Loop through each site
    for site in sites:
        # Loop through each pollutant
        for pollutant in pollutants:
            # Create a new figure for each site and pollutant
            fig, axs = plt.subplots(4, 2, figsize=(16, 16), dpi=dpi)
            fig.suptitle(f'Diurnal Trends of {pollutant} at {site}', fontsize=16)
            
            # Flatten the axs array for easier iteration
            axs = axs.flatten()
            
            # Loop through each day
            for i, day in enumerate(days):
                # Set the current subplot
                ax = axs[i]
                
                # Filter the DataFrame for the current day, pollutant, and site
                df_site_pollutant_day = df_melted[(df_melted['Pollutant'] == pollutant) & 
                                                   (df_melted['Site'] == site) & 
                                                   (df_melted['Day'] == day)]
                
                # Plot the diurnal trends on the current subplot
                sns.lineplot(data=df_site_pollutant_day, x='Hour', y='Value', 
                             marker='o', dashes=False, errorbar=None, ax=ax)
                
                # Calculate 95% confidence interval
                sns.lineplot(data=df_site_pollutant_day, x='Hour', y='Value', 
                             estimator='mean', ci=95, color='blue', ax=ax)
                
                # Set title, x-label, and y-label for the current subplot
                ax.set_title(day)
                ax.set_xlabel('Hour of the Day')
                ax.set_ylabel(f'Concentration ({pollutant})')
                
                # Adjust layout
                plt.tight_layout()
            
            # Turn off the last subplot (4, 2)
            axs[-1].set_axis_off()
    
            # Show plot for the current site and pollutant
            plt.show()

# Assuming 'df_melted' is your melted DataFrame
plot_diurnal_trends_per_site_and_pollutant_per_day(df_melted)

# Function to create boxplots for each pollutant with 'Site' on x-axis, 'Value' on y-axis, and 'Day_Category' as hue
def plot_weekday_weekend_comparison(df, dpi=200):
    pollutants = df['Pollutant'].unique()
    
    for pollutant in pollutants:
        df_pollutant = df[df['Pollutant'] == pollutant]
        
        plt.figure(figsize=(15, 5), dpi=dpi)
        sns.boxplot(data=df_pollutant, x='Site', y='Value', hue='Day_Category', showfliers=False)
              
        plt.title(f'Weekday vs. Weekend Concentrations of {pollutant}')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.xlabel('')
        plt.ylabel(f'Concentration ({units[pollutant]})')
        plt.legend(title='Day Category', loc='upper right')
        plt.tight_layout()
        plt.show()

# Call the function to create the plots
plot_weekday_weekend_comparison(df_melted)

def plot_seasonal_comparison(df, dpi=200):
    pollutants = df['Pollutant'].unique()
    
    for pollutant in pollutants:
        df_pollutant = df[df['Pollutant'] == pollutant]
        
        plt.figure(figsize=(15, 5), dpi=dpi)
        sns.boxplot(data=df_pollutant, x='Site', y='Value', hue='Season', showfliers=False)
              
        plt.title(f'Seasonal Concentrations of {pollutant}')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.xlabel('')
        plt.ylabel(f'Concentration ({units[pollutant]})')
        plt.legend(title='Season', loc='upper right')
        plt.tight_layout()
        plt.show()

# Call the function to create the plots
plot_seasonal_comparison(df_melted)

def plot_seasonal_comparison_per_site(df, dpi=200):
    pollutants = df['Pollutant'].unique()
    sites = df['Site'].unique()
    
    for pollutant in pollutants:
        for site in sites:
            df_site_pollutant = df[(df['Pollutant'] == pollutant) & (df['Site'] == site)]
            
            # Check if the DataFrame is not empty
            if not df_site_pollutant.empty:
                plt.figure(figsize=(5, 5), dpi=dpi)
                sns.boxplot(data=df_site_pollutant, x='Season', y='Value', showfliers=False)
                
                plt.title(f'Seasonal Concentrations of {pollutant} at {site}')
                plt.xlabel('Season')
                plt.ylabel(f'Concentration ({units[pollutant]})')
                plt.tight_layout()
                plt.show()

# Call the function to create the plots
plot_seasonal_comparison_per_site(df_melted)

def plot_monthly_comparison(df, dpi=200):
    pollutants = df['Pollutant'].unique()
    
    for pollutant in pollutants:
        df_pollutant = df[df['Pollutant'] == pollutant]
        
        for month in df_pollutant['Month'].unique():
            df_pollutant_month = df_pollutant[df_pollutant['Month'] == month]
            
            # Check if the DataFrame is not empty
            if not df_pollutant_month.empty:
                plt.figure(figsize=(20, 5), dpi=dpi)
                sns.boxplot(data=df_pollutant_month, x='Site', y='Value', hue='Month', showfliers=False)
                
                plt.title(f'Monthly Concentrations of {pollutant} for {month}')
                plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
                plt.xlabel('')
                plt.ylabel(f'Concentration ({units[pollutant]})')
                plt.legend(title='Month', loc='upper right')
                plt.tight_layout()
                plt.show()

# Call the function to create the plots
plot_monthly_comparison(df_melted)

def plot_monthly_comparison_per_site(df, dpi=200):
    pollutants = df['Pollutant'].unique()
    sites = df['Site'].unique()
    
    for pollutant in pollutants:
        for site in sites:
            df_site_pollutant = df[(df['Pollutant'] == pollutant) & (df['Site'] == site)]
            
            # Check if the DataFrame is not empty
            if not df_site_pollutant.empty:
                plt.figure(figsize=(6, 5), dpi=dpi)
                sns.boxplot(data=df_site_pollutant, x='Month', y='Value', showfliers=False)
                
                plt.title(f'Monthly Concentrations of {pollutant} at {site}')
                plt.xlabel('Month')
                plt.ylabel(f'Concentration ({units[pollutant]})')
                plt.tight_layout()
                plt.show()

# Call the function to create the plots
plot_monthly_comparison_per_site(df_melted)

def plot_overall(df, dpi=200):
    pollutants = df['Pollutant'].unique()
    
    for pollutant in pollutants:
        df_pollutant = df[df['Pollutant'] == pollutant]
        
        plt.figure(figsize=(15, 5), dpi=dpi)
        sns.boxplot(data=df_pollutant, x='Site', y='Value', showfliers=False)
              
        plt.title(f'Overall Concentrations of {pollutant}')
        plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
        plt.xlabel('')
        plt.ylabel(f'Concentration ({units[pollutant]})')
        plt.tight_layout()
        plt.show()

# Call the function to create the plots
plot_overall(df_melted)

# Create a heatmap of Spearman correlation
o3_df = consolidated_df[['O3', 'Site']].copy()
co_df = consolidated_df[['CO', 'Site']].copy()
pm25_df = consolidated_df[['PM2.5', 'Site']].copy()

# Reset the index for each dataframe to make datetime a column again
o3_df.reset_index(inplace=True)
co_df.reset_index(inplace=True)
pm25_df.reset_index(inplace=True)

def pivot_dataframe(df):
    # Aggregate duplicate entries by taking the mean for each datetime index and site
    df_aggregated = df.groupby(['datetime', 'Site']).mean().reset_index()
    # Pivot the aggregated DataFrame
    df_pivot = df_aggregated.pivot(index='datetime', columns='Site', values=df.columns[1])
    return df_pivot

# Create pivoted DataFrames for each pollutant
o3_pivot = pivot_dataframe(o3_df)
co_pivot = pivot_dataframe(co_df)
pm25_pivot = pivot_dataframe(pm25_df)

# Create a function to plot heatmap correlation matrix with Spearman correlation
def plot_heatmap_correlation_spearman(df, title, dpi=200):
    plt.figure(figsize=(25, 10))
    mask = np.triu(np.ones_like(df.corr(method='spearman'), dtype=bool))
    sns.heatmap(df.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f", mask=mask, cbar_kws={'label': 'Spearman correlation coefficient'}) # Remove method to use Pearson
    plt.title(title)
    plt.xlabel('')
    plt.show()

# Plot heatmap correlation matrix with Spearman correlation for each pollutant
plot_heatmap_correlation_spearman(o3_pivot, 'Ozone Spearman Correlation Matrix')
plot_heatmap_correlation_spearman(co_pivot, 'CO Spearman Correlation Matrix')
plot_heatmap_correlation_spearman(pm25_pivot, 'PM2.5 Spearman Correlation Matrix')

# Function to calculate Jensen-Shannon Divergence
def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd

# Function to calculate coefficient of divergence
def coefficient_of_divergence(jsd_matrix):
    max_jsd = np.max(jsd_matrix)
    min_jsd = np.min(jsd_matrix)
    return (jsd_matrix - min_jsd) / (max_jsd - min_jsd)

# Function to compute the matrix of Jensen-Shannon Divergence values
def compute_jsd_matrix(data):
    n_sites = data.shape[1]
    jsd_matrix = np.zeros((n_sites, n_sites))
    for i in range(n_sites):
        for j in range(n_sites):
            jsd_matrix[i, j] = jensen_shannon_divergence(data.iloc[:, i], data.iloc[:, j])
    return jsd_matrix

# Function to plot heatmap of coefficient of divergence
def plot_cod_heatmap(data, site_labels, title, dpi=200):
    cod_matrix = coefficient_of_divergence(data)
    mask = np.triu(np.ones_like(cod_matrix))  # Create mask for upper triangular part
    plt.figure(figsize=(25, 10))
    sns.heatmap(cod_matrix, mask=mask, annot=True, cmap='viridis', fmt=".2f", cbar_kws={'label': 'Coefficient of Divergence'})
    plt.title(title)
    plt.xlabel('')
    plt.ylabel('Sites')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)
    plt.xticks(range(len(site_labels)), site_labels)  # Set x-axis ticks and labels
    plt.yticks(range(len(site_labels)), site_labels)  # Set y-axis ticks and labels
    plt.tight_layout()
    plt.show()

# Compute the Jensen-Shannon Divergence matrices for CO and PM2.5
o3_pivot_columns = o3_pivot.columns.tolist()  # Get column names as list
co_pivot_columns = co_pivot.columns.tolist()  # Get column names as list
pm25_pivot_columns = pm25_pivot.columns.tolist()  # Get column names as list

jsd_matrix = compute_jsd_matrix(o3_pivot)
co_jsd_matrix = compute_jsd_matrix(co_pivot)
pm25_jsd_matrix = compute_jsd_matrix(pm25_pivot)

# Plot heatmap of coefficient of divergence
plot_cod_heatmap(jsd_matrix, o3_pivot_columns, 'Coefficient of Divergence Heatmap (O3)')

# Plot heatmap of coefficient of divergence for CO
plot_cod_heatmap(co_jsd_matrix, co_pivot_columns, 'Coefficient of Divergence Heatmap (CO)')

# Plot heatmap of coefficient of divergence for PM2.5
plot_cod_heatmap(pm25_jsd_matrix, pm25_pivot_columns, 'Coefficient of Divergence Heatmap (PM2.5)')
