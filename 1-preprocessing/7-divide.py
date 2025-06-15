#!/usr/bin/env python3

'''
Divide each currency exchange rate time series dataset into 3 subsets:
- training (70%)
- cross-validation (20%)
- testing (10%)
'''

import os
import pandas as pd
from sklearn.model_selection import train_test_split

input_dir = 'data/6-time-lags/USD'
output_dir = 'data/7-subsets/USD'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through the CSV files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        input_file = os.path.join(input_dir, filename)

        # Load data
        data = pd.read_csv(input_file)
        X = data.iloc[:, :-1]  # Feature data
        y = data.iloc[:, -1]   # Target data

        # Split data sequentially into training, cross-validation, and testing sets
        X_train, X_cv_test, y_train, y_cv_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        X_cv, X_test, y_cv, y_test = train_test_split(X_cv_test, y_cv_test, test_size=2/3, shuffle=False)

        # Combine the X and y data for each split
        train_data = pd.concat([X_train, y_train], axis=1)
        cv_data = pd.concat([X_cv, y_cv], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        # Create the output directory for the current file
        file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        # Save the combined data to CSV files
        train_data.to_csv(os.path.join(file_output_dir, 'train_data.csv'), index=False)
        cv_data.to_csv(os.path.join(file_output_dir, 'cv_data.csv'), index=False)
        test_data.to_csv(os.path.join(file_output_dir, 'test_data.csv'), index=False)

