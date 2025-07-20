#!/usr/bin/env python3

# Reverse transform the forecast data

import os
import pandas as pd
import numpy as np
import datetime

# Load data
log_org = pd.read_csv('../../1-preprocessing/data/decorrelation-tests/1-correlated/3-log-transformed.csv')
test_set = pd.read_csv('../../1-preprocessing/data/decorrelation-tests/5-split/log-differenced/USD/5/test_data.csv')

log_org['Date'] = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in log_org['Date']]
test_set['Date'] = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in test_set['Date']]

first_day = test_set['Date'].iloc[0]
prev_day = first_day - pd.Timedelta(days=1)

input_dir = 'data'

for file in os.listdir(input_dir):
    if 'forecast.csv' in file:
        print(file)
        data = pd.read_csv(os.path.join(input_dir, file))
        data['Date'] = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in data['Date']]
        
        # Reverse difference y_test and y_pred
        prev_log = log_org.loc[log_org['Date'] == prev_day, 'USD'].values[0]
        first = {'Date': pd.Series(prev_day),
                 'y_test': pd.Series(prev_log),
                 'y_pred': pd.Series(prev_log)}
        first_df = pd.DataFrame(first)
        rev = pd.concat([first_df, data], axis=0)
        print(rev)
        rev['y_test'] = rev['y_test'].cumsum()
        rev['y_pred'] = rev['y_pred'].cumsum()

        # Reverse log transform
        rev['y_test'] = np.exp(rev['y_test'])
        rev['y_pred'] = np.exp(rev['y_pred'])

        # Save the data to output file
        output_file = file.replace('forecast', 'forecast_reverse')
        output_path = os.path.join(input_dir, output_file)
        rev.to_csv(output_path, index=False)

