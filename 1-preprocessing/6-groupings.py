#!/usr/bin/env python3

'''Split data into individual currency time series groupings.'''

import os
import pandas as pd

# Set the input and output directories
input_file = 'data/5-normalized.csv'
output_dir = 'data/6-groupings'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
df = pd.read_csv(input_file, index_col='Date')

# Define the groupings
groupings = {
    'HUF': ['ZAR', 'CHF', 'NOK', 'SEK', 'SGD'],
    'ZAR': ['HUF', 'CHF', 'NOK', 'SEK', 'GBP'],
    'CHF': ['HUF', 'ZAR', 'SGD', 'NOK', 'GBP'],
    'NOK': ['HUF', 'SEK', 'ZAR', 'CHF', 'SGD'],
    'SEK': ['NOK', 'ZAR', 'HUF', 'GBP', 'CHF'],
    'GBP': ['CZK', 'ZAR', 'CHF', 'HUF', 'KRW'],
    'SGD': ['CHF', 'HUF', 'ZAR', 'USD', 'HKD'],
    'CZK': ['GBP', 'ZAR', 'CHF', 'HUF', 'KRW'],
    'HKD': ['USD', 'KRW', 'SGD', 'CZK', 'GBP'],
    'USD': ['HKD', 'KRW', 'SGD', 'CZK', 'GBP'],
    'NZD': ['AUD', 'CHF', 'CZK', 'SGD', 'ZAR'],
    'KRW': ['USD', 'HKD', 'GBP', 'CZK', 'SEK'],
    'PLN': ['HUF', 'CHF', 'NOK', 'ZAR', 'GBP'],
    'AUD': ['NZD', 'CAD', 'CHF', 'CZK', 'SGD'],
    'CAD': ['AUD', 'SGD', 'NZD', 'JPY', 'CHF'],
    'JPY': ['SEK', 'NOK', 'ZAR', 'CAD', 'CZK'],
    'DKK': ['CZK', 'PLN', 'ZAR', 'JPY', 'NOK']
}

# Process each grouping and save the CSV files
for currency, columns in groupings.items():
    output_file = os.path.join(output_dir, f'{currency}.csv')
    output_columns = columns + [currency]
    df[output_columns].to_csv(output_file)

