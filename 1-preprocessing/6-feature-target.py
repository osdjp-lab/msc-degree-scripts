#!/usr/bin/env python3

'''Create 1-50 time lagged (windowed) datasets from the split datasets.'''

import os
import pandas as pd

# Set the input and output directories
input_dir = 'data/5-split'
output_dir = 'data/6-time-lags'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through the files in the input directory
for filename in os.listdir(input_dir):
    # if filename == "USD.csv":   # Comment for all files in input directory
        if filename.endswith('.csv'):
            # Load the data from the input file
            input_file = os.path.join(input_dir, filename)
            df = pd.read_csv(input_file)
            value_column = os.path.splitext(filename)[0]

            # Create the output directory for the current file
            file_output_dir = os.path.join(output_dir, value_column)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)

            # Create 1 to 50 offset variations
            for offset in range(1, 51):
                # Create a new DataFrame for the current offset
                offset_df = df.copy()

                # Create the new columns
                for i in range(1, offset + 1):
                    offset_df[f'{value_column}_shifted_{i}'] = offset_df[value_column].shift(i)

                # Drop rows with NaN or null values
                offset_df = offset_df.dropna(subset=[f'{value_column}_shifted_{i}' for i in range(1, offset + 1)])

                # Save the data to a CSV file
                output_file = os.path.join(file_output_dir, f'{offset}.csv')
                columns_to_save = ['Date'] + [f'{value_column}_shifted_{i}' for i in range(1, offset + 1)] + [value_column]
                offset_df[columns_to_save].to_csv(output_file, index=False)

