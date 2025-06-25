#!/usr/bin/env python3

'''Create 1 to nr_lags time lagged (windowed) datasets from the split datasets.'''

import os
import pandas as pd

# Set the input and output directories
input_dir = 'data/7-groupings'
output_dir = 'data/8-time-lags'

nr_lags = 150

# Create output directory if not exists
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file
for file in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:
    # if file == 'USD.csv':
        df = pd.read_csv(os.path.join(input_dir, file))
        base_name = os.path.splitext(file)[0]

        # Get actual column data references
        date_col = df.columns[0]  # Name of date column
        predictors = df.columns[1:-1]
        target_col = df.columns[-1]  # Name of target column

        # Initialize with date column DATA (not just name)
        shifted_data = df[[date_col]].copy()
        
        # Create shifted predictors
        for shift in range(1, nr_lags+1):
            shifted = df[predictors].shift(shift)
            shifted.columns = [f'{col}_shift{shift}' for col in predictors]
            shifted_data = pd.concat([shifted_data, shifted], axis=1)
        
        # Add target column DATA (not just name)
        shifted_data[target_col] = df[target_col]
        
        # Clean and save
        shifted_data.dropna().to_csv(
            os.path.join(output_dir, f'{base_name}.csv'),
            index=False
        )

