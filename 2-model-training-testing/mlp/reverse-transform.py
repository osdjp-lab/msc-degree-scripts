#!/usr/bin/env python3

# Reverse transform the forecast data

import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler

# Load data
raw = pd.read_csv('../../1-preprocessing/data/decorrelation-tests/4-features/raw/USD/7.csv')
diff_org = pd.read_csv('../../1-preprocessing/data/decorrelation-tests/1-correlated/3-differenced.csv')
test_set = pd.read_csv('../../1-preprocessing/data/decorrelation-tests/5-split/diff-normalized/USD/7/test_data.csv')

raw['Date'] = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in raw['Date']]
test_set['Date'] = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in test_set['Date']]

first_day = test_set['Date'].iloc[0]
prev_day = first_day - pd.Timedelta(days=1)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(diff_org[['USD']])

input_dir = 'data'

for file in os.listdir(input_dir):
    if 'forecast.csv' in file:
        print(file)
        data = pd.read_csv(os.path.join(input_dir, file))
        data['Date'] = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in data['Date']]
        
        # Reverse normalize y_test and y_pred
        data['y_test'] = scaler.inverse_transform(data[['y_test']])
        data['y_pred'] = scaler.inverse_transform(data[['y_pred']])

        # Reverse difference y_test and y_pred
        prev_raw = raw.loc[raw['Date'] == prev_day, 'USD'].values[0]
        first = {'Date': pd.Series(prev_day),
                 'y_test': pd.Series(prev_raw),
                 'y_pred': pd.Series(prev_raw)}
        first_df = pd.DataFrame(first)
        rev = pd.concat([first_df, data], axis=0)
        print(rev)
        rev['y_test'] = rev['y_test'].cumsum()
        rev['y_pred'] = rev['y_pred'].cumsum()

        # Save the data to output file
        output_file = file.replace('forecast', 'forecast_reverse')
        output_path = os.path.join(input_dir, output_file)
        rev.to_csv(output_path, index=False)

