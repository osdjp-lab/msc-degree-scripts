#!/usr/bin/env python3

'''Select suitable variables from the initial dataset.'''

import os
import pandas as pd

input_file = 'data/0-raw.csv'
output_file = 'data/1-selected.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Create the output directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Drop columns with missing values
df_selected = df.dropna(axis=1, how='any')

# Save the selected data to a new CSV file
df_selected.to_csv(output_file, index=False)

