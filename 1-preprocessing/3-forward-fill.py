#!/usr/bin/env python3

'''Fill in missing dates and forward fill missing values.'''

import pandas as pd

input_file = 'data/2-decorrelated.csv'
output_file = 'data/3-filled.csv'

# Load the CSV file
data = pd.read_csv(input_file, index_col='Date', parse_dates=True)

# Create a complete date range
start_date = data.index.min()
end_date = data.index.max()
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Reindex the DataFrame
data_reindexed = data.reindex(date_range)

# Forward fill the values
data_forward_filled = data_reindexed.ffill()

# Save the updated DataFrame to a CSV file
data_forward_filled.to_csv(output_file, index_label='Date')

