#!/usr/bin/env python3

# Reverse transform the forecast data for the test set

import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler

currency = '12-USD'

# # # # # # # # # #

def rev_diff(initial, differenced):
    first_date = differenced['Date'].iloc[0]
    
    matching_indices = initial.index[initial['Date'] == first_date].tolist()
    if matching_indices:
        target_idx = matching_indices[0]
        prev_idx = target_idx - 1
        prev_date = initial.at[prev_idx, 'Date']
        prev_value = initial.at[prev_idx, currency]
    
    first = {'Date': pd.Series(prev_date),
             'y_test': pd.Series(prev_value),
             'y_pred': pd.Series(prev_value)}
    first_df = pd.DataFrame(first)

    rev = pd.concat([first_df, differenced], axis=0)
    rev['y_test'] = rev['y_test'].cumsum()
    rev['y_pred'] = rev['y_pred'].cumsum()

    return rev

def rev_log(log_transformed):
    
    log_transformed['y_test'] = np.exp(log_transformed[['y_test']])
    log_transformed['y_pred'] = np.exp(log_transformed[['y_pred']])
    
    return log_transformed

def rev_norm(un_norm, norm):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(un_norm[[currency]])
    
    norm['y_test'] = scaler.inverse_transform(norm[['y_test']])
    norm['y_pred'] = scaler.inverse_transform(norm[['y_pred']])
    
    return norm

def rev_std(un_std, std):
    scaler = StandardScaler()
    scaler.fit(un_std[[currency]])
    
    std['y_test'] = scaler.inverse_transform(std[['y_test']])
    std['y_pred'] = scaler.inverse_transform(std[['y_pred']])
    
    return std

# # # # # # # # # #

mapping = {'d' : 'differenced',
           'dn' : 'diff-normalized',
           'ds' : 'diff-standardized',
           'ld' : 'log-differenced',
           'ldn' : 'log-diff-normalized',
           'lds' : 'log-diff-standardized',
           'ln' : 'log-normalized',
           'ls' : 'log-standardized',
           'l' : 'log-transformed',
           'n' : 'normalized',
           'r' : 'raw',
           's' : 'standardized'}

for model in ('mlp', 'svr', 'rf'):

    input_dir = f'../data/3-training-testing/{model}'

    selection_dir = f"../data/2-window-selection/2-optimal/{model}/mse"
    mse_results = pd.read_csv(os.path.join(selection_dir, "min_avg_mse_results.csv"))
    min_idx = mse_results['mse'].idxmin()
    optimal_offset_dataset = mse_results.loc[min_idx, 'method']
    method, offset = optimal_offset_dataset.split('-')
    
    for input_file in os.listdir(input_dir):
        if 'forecast.csv' in input_file:
            input_path = os.path.join(input_dir, input_file)
            output_file = input_file.replace('forecast', 'forecast_reverse')
            output_path = os.path.join(input_dir, output_file)
            print(input_path)
            test_set_forecast = pd.read_csv(input_path)

            if method == 'd':
                ref_dataset = f'../data/1-preprocessing/4-decorrelated/raw/{currency}.csv'

                undifferenced_unsplit = pd.read_csv(ref_dataset)
                
                r_diff = rev_diff(undifferenced_unsplit, test_set_forecast)
                r_diff.to_csv(output_path, index=False)
            
            if method == 'dn':
                ref_dataset_1 = f'../data/1-preprocessing/4-decorrelated/raw/{currency}.csv'
                ref_dataset_2 = f'../data/1-preprocessing/4-decorrelated/differenced/{currency}.csv'
                
                un_norm = pd.read_csv(ref_dataset_2)
                undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                
                r_norm = rev_norm(un_norm, test_set_forecast)
                r_norm_diff = rev_diff(undifferenced_unsplit, r_norm)
                r_norm_diff.to_csv(output_path, index=False)
            
            if method == 'ds':
                ref_dataset_1 = f'../data/1-preprocessing/4-decorrelated/raw/{currency}.csv'
                ref_dataset_2 = f'../data/1-preprocessing/4-decorrelated/differenced/{currency}.csv'
                
                un_std = pd.read_csv(ref_dataset_2)
                undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                
                r_std = rev_std(un_std, test_set_forecast)
                r_std_diff = rev_diff(undifferenced_unsplit, r_std)
                r_std_diff.to_csv(output_path, index=False)
            
            if method == 'ld':
                ref_dataset = f'../data/1-preprocessing/4-decorrelated/log-transformed/{currency}.csv'
                
                undifferenced_unsplit = pd.read_csv(ref_dataset)

                r_diff = rev_diff(undifferenced_unsplit, test_set_forecast)
                r_diff_log = rev_log(r_diff)
                r_diff_log.to_csv(output_path, index=False)
            
            if method == 'ldn':
                ref_dataset_1 = f'../data/1-preprocessing/4-decorrelated/log-transformed/{currency}.csv'
                ref_dataset_2 = f'../data/1-preprocessing/4-decorrelated/log-differenced/{currency}.csv'
                
                un_norm = pd.read_csv(ref_dataset_2)
                undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                
                r_norm = rev_norm(un_norm, test_set_forecast)
                r_norm_diff = rev_diff(undifferenced_unsplit, r_norm)
                r_norm_diff_log = rev_log(r_norm_diff)
                r_norm_diff_log.to_csv(output_path, index=False)
            
            if method == 'lds':
                ref_dataset_1 = f'../data/1-preprocessing/4-decorrelated/log-transformed/{currency}.csv'
                ref_dataset_2 = f'../data/1-preprocessing/4-decorrelated/log-differenced/{currency}.csv'
                
                un_std = pd.read_csv(ref_dataset_2)
                undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                
                r_std = rev_std(un_std, test_set_forecast)
                r_std_diff = rev_diff(undifferenced_unsplit, r_std)
                r_std_diff_log = rev_log(r_std_diff)
                r_std_diff_log.to_csv(output_path, index=False)
            
            if method == 'ln':
                ref_dataset = f'../data/1-preprocessing/4-decorrelated/log-transformed/{currency}.csv'
                
                un_norm = pd.read_csv(ref_dataset)
                
                r_norm = rev_norm(un_norm, test_set_forecast)
                r_norm_log = rev_log(r_norm)
                r_norm_log.to_csv(output_path, index=False)
            
            if method == 'ls':
                ref_dataset = f'../data/1-preprocessing/4-decorrelated/log-transformed/{currency}.csv'
                
                un_std = pd.read_csv(ref_dataset)
                
                r_std = rev_std(un_std, test_set_forecast)
                r_std_log = rev_log(r_std)
                r_std_log.to_csv(output_path, index=False)
            
            if method == 'l':
                r_log = rev_log(test_set_forecast)
                r_log.to_csv(output_path, index=False)
            
            if method == 'n':
                ref_dataset = f'../data/1-preprocessing/4-decorrelated/raw/{currency}.csv'

                un_norm = pd.read_csv(ref_dataset)
                
                r_norm = rev_norm(un_norm, test_set_forecast)
                r_norm.to_csv(output_path, index=False)
            
            if method == 'r':
                test_set_forecast.to_csv(output_path, index=False)
            
            if method == 's':
                ref_dataset = f'../data/1-preprocessing/4-decorrelated/raw/{currency}.csv'

                un_std = pd.read_csv(ref_dataset)
                
                r_std = rev_std(un_std, test_set_forecast)
                r_std.to_csv(output_path, index=False)

