#!/usr/bin/env python3

'''Calculate the correlation matrix for each currency exchange rate.'''

import os
import pandas as pd
import numpy as np

input_file = 'data/1-selected.csv'
output_file = 'data/2-decorrelated.csv'

full_correlation_matrix_file = 'meta/full_correlation_matrix.csv'
half_correlation_matrix_file = 'meta/half_correlation_matrix.csv'
reduced_full_correlation_matrix_file = 'meta/reduced_full_correlation_matrix.csv'
reduced_half_correlation_matrix_file = 'meta/reduced_half_correlation_matrix.csv'

high_correlations_file = 'meta/high_correlation_pairs.csv'
low_correlations_file = 'meta/low_correlation_pairs.csv'

# Load the CSV file
df = pd.read_csv(input_file)

# Create the output directory if it doesn't exist
os.makedirs('meta', exist_ok=True)

##########################
# Full correlation matrix
##########################

# Calculate the Pearson correlation matrix
corr_matrix = df.iloc[:, 1:].corr(method='pearson')

# Save the full correlation matrix to a new CSV file
corr_matrix.to_csv(full_correlation_matrix_file)

##########################
# Half correlation matrix
##########################

# Create a lower triangular mask
mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

# Save the lower triangle of the correlation matrix to a new CSV file
half_corr_matrix = corr_matrix.where(mask)
half_corr_matrix.to_csv(half_correlation_matrix_file)

#########################
# High correlation pairs
#########################

# Find the currency pairs with correlation above 0.8 or below -0.8
high_corr_pairs = corr_matrix[(corr_matrix > 0.8) | (corr_matrix < -0.8)].stack().reset_index()
high_corr_pairs.columns = ['Currency 1', 'Currency 2', 'Correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'] != 1.0]  # Exclude the diagonal (self-correlation)

# Save the high_corr_pairs to a CSV file
high_corr_pairs.to_csv(high_correlations_file, index=False)

#######################
# Decorrelated dataset
#######################

# Remove selected highly correlated (> 0.8) columns from the dataset

# List of columns to drop
columns_to_drop = ['CZK', 'HUF', 'NOK', 'HKD', 'SGD', 'ZAR']

# Drop the selected columns
df_cleaned = df.drop(columns_to_drop, axis=1)

# Save the cleaned dataset to a new CSV file
df_cleaned.to_csv(output_file, index=False)

##################################
# Reduced full correlation matrix
##################################

# Calculate the Pearson correlation matrix
red_corr_matrix = df_cleaned.iloc[:, 1:].corr(method='pearson')

# Save the full correlation matrix to a new CSV file
red_corr_matrix.to_csv(reduced_full_correlation_matrix_file)

##################################
# Reduced half correlation matrix
##################################

# Create a lower triangular mask
mask = np.tril(np.ones_like(red_corr_matrix, dtype=bool))

# Save the lower triangle of the correlation matrix to a new CSV file
half_red_corr_matrix = red_corr_matrix.where(mask)
half_red_corr_matrix.to_csv(reduced_half_correlation_matrix_file)

###############
# Correlations
###############

# Create the destination directory if it doesn't exist
dest_dir = 'meta/correlations'
os.makedirs(dest_dir, exist_ok=True)

# Loop through each column in the correlation matrix
for col in red_corr_matrix.columns:
    # Get the correlation values for the current column
    col_corr = red_corr_matrix[col].abs().sort_values()
    
    # Create a new DataFrame with the sorted correlation values
    col_df = pd.DataFrame({col: col_corr})
    
    # Save the DataFrame to a CSV file in the destination directory
    col_df.to_csv(os.path.join(dest_dir, f"{col}.csv"), index=True)

#####################
# Low correlations
#####################

# Initialize an empty DataFrame to store the results
composite_low_5 = pd.DataFrame(columns=['Currency', 'Low Correlations Sum'])

# Loop through each CSV file in the directory
for filename in os.listdir(dest_dir):
    if filename.endswith('.csv'):
        # Load the CSV file
        df = pd.read_csv(os.path.join(dest_dir, filename))
        
        # Get the low 5 absolute values and their sum
        low_5 = df.iloc[1:5, 1].sum()
        currency = filename.split('.')[0]

        # Create a new row and concatenate it to the composite DataFrame
        new_row = pd.DataFrame({'Currency': [currency], 'Low Correlations Sum': [low_5]})
        composite_low_5 = pd.concat([composite_low_5, new_row], ignore_index=True)

# Sort the composite DataFrame in descending order by 'Low 5 Correlations Sum'
composite_low_5 = composite_low_5.sort_values(by='Low Correlations Sum', ascending=True)

# Save the composite DataFrame to a new CSV file
composite_low_5.to_csv(low_correlations_file, index=False)

