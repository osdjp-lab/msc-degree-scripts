#!/usr/bin/env python3

'''Split data into individual currency time series groupings.'''

import os
import pandas as pd

# Set the input and output directories
input_file = 'data/6-normalized.csv'
output_dir = 'data/7-groupings'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(input_file, index_col='Date')

# Define the groupings
groupings = {
    'AUD': ['KRW', 'DKK', 'PLN', 'USD', 'JPY'],
    'CAD': ['GBP', 'USD', 'DKK', 'PLN', 'SEK'],
    'CHF': ['JPY', 'KRW', 'DKK', 'USD', 'CAD'],
    'DKK': ['KRW', 'AUD', 'NZD', 'USD', 'GBP'],
    'GBP': ['CAD', 'JPY', 'DKK', 'AUD', 'USD'],
    'JPY': ['PLN', 'NZD', 'CHF', 'GBP', 'DKK'],
    'KRW': ['AUD', 'DKK', 'NZD', 'CHF', 'PLN'],
    'NZD': ['JPY', 'SEK', 'USD', 'DKK', 'KRW'],
    'PLN': ['JPY', 'AUD', 'USD', 'KRW', 'CAD'],
    'SEK': ['NZD', 'USD', 'DKK', 'CAD', 'AUD'],
    'USD': ['NZD', 'SEK', 'CAD', 'DKK', 'PLN']
}

# Process each grouping and save the CSV files
for currency, columns in groupings.items():
    output_file = os.path.join(output_dir, f'{currency}.csv')
    output_columns = columns + [currency]
    df[output_columns].to_csv(output_file)

