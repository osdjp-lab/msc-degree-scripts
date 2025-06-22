#!/usr/bin/env python3

'''Apply normalization to the data.'''

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_file  = 'data/5-differenced.csv'
output_file = 'data/6-normalized.csv'

# Load the CSV file
data = pd.read_csv(input_file, index_col='Date')

# Normalize the differenced data, excluding the header
header = data.columns
data_values = data.values
scaler = MinMaxScaler(feature_range=(-1,1))
data_normalized = scaler.fit_transform(data_values)

# Convert the normalized data back to a DataFrame
data_normalized_df = pd.DataFrame(data_normalized, index=data.index, columns=header)

# Save the normalized DataFrame to a CSV file
data_normalized_df.to_csv(output_file, index=True, header=True)

