#!/usr/bin/env python3

'''Check stationarity and apply log differencing.'''

import pandas as pd
from statsmodels.tsa.stattools import adfuller

input_file = 'data/4-log-transformed.csv'
output_file = 'data/5-differenced.csv'

# Load the CSV file
data = pd.read_csv(input_file, index_col='Date')

print("--------------------------")
print("ADF for undifferenced data")
print("--------------------------")
print("")

# Perform ADF test on each column
for column in data.columns:

    # Perform ADF test
    adf_result = adfuller(data[column])
    
    # Print the ADF test results
    print('ADF Statistic: ', adf_result[0])
    print('p-value: ', adf_result[1])
    print('Critical Values: ')
    for key, value in adf_result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # Determine stationarity
    if adf_result[1] < 0.05:
        print(f"{column}: stationary")
    else:
        print(f"{column}: non-stationary")
    print("-----------------------------------")

data_diff = data

if True:

    # Perform differencing
    data_diff = data.diff()
    
    # Drop the first row which will have NaN values after differencing
    data_diff = data_diff.dropna()
    
    print("")
    print("------------------------")
    print("ADF for differenced data")
    print("------------------------")
    print("")
    
    # Perform ADF test on each column of the differenced data
    for column in data_diff.columns:
    
        # Perform ADF test
        adf_result = adfuller(data_diff[column])
        
        # Print the ADF test results
        print('ADF Statistic: ', adf_result[0])
        print('p-value: ', adf_result[1])
        print('Critical Values: ')
        for key, value in adf_result[4].items():
            print('\t%s: %.3f' % (key, value))
        
        # Determine stationarity
        if adf_result[1] < 0.05:
            print(f"{column}: stationary")
        else:
            print(f"{column}: non-stationary")
        print("-----------------------------------")
    
# Reset the index to include the Date column
data_diff_reset = data_diff.reset_index()

# Save the differenced DataFrame to a CSV file
data_diff_reset.to_csv(output_file, index=False)

