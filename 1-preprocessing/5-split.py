#!/usr/bin/env python3

'''Split data into individual currency time series.'''

import os
import pandas as pd

# Set the input and output directories
input_file = 'data/4-normalized.csv'
output_dir = 'data/5-split'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV file
df = pd.read_csv(input_file)

# Loop through the columns (excluding the first column)
for col in df.columns[1:]:
    # Create a new DataFrame with the index and the current column
    new_df = df[['Date', col]]
    
    # Save the new DataFrame to a CSV file in the output directory
    new_df.to_csv(os.path.join(output_dir, f'{col}.csv'), index=False)

