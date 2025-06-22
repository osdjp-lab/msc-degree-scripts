#!/usr/bin/env python3

'''Apply a log transformation to the data.'''

import pandas as pd
import numpy as np

input_file = 'data/3-filled.csv'
output_file = 'data/4-log-transformed.csv'

# Load the CSV file
df = pd.read_csv(input_file, index_col='Date')

df_log = df

# Apply log transformation
df_log = df.apply(lambda x: pd.Series(np.log(x)))

# Save the normalized DataFrame to a CSV file
df_log.to_csv(output_file, index=True, header=True)

