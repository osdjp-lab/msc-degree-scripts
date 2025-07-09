#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

input_dir = '../data/rf'
output_dir = 'data/rf/mse'
os.makedirs(output_dir, exist_ok=True)

# MSE

train_mse_results = pd.DataFrame(columns=['method', 'mse'])
test_mse_results = pd.DataFrame(columns=['method', 'mse'])
min_avg_mse_results = pd.DataFrame(columns=['method', 'mse'])

for subdir in os.listdir(input_dir):
        for target in os.listdir(os.path.join(input_dir, subdir)):
            rel_path = os.path.join(subdir, target)
            
            print(rel_path)
            
            metric = 'mse'
            
            train_data = pd.read_csv(os.path.join(input_dir, rel_path,f"train_{metric}_results.csv")).sort_values(by='offset')
            test_data = pd.read_csv(os.path.join(input_dir, rel_path, f"test_{metric}_results.csv")).sort_values(by='offset')

            min_train_idx = train_data[metric].idxmin()
            min_train_offset = train_data.loc[min_train_idx, 'offset'] 
            min_train_value = train_data.loc[min_train_idx, metric] 
            
            min_test_idx = test_data[metric].idxmin()
            min_test_offset = test_data.loc[min_test_idx, 'offset'] 
            min_test_value = test_data.loc[min_test_idx, metric] 

            combined_avg = (train_data[metric] + test_data[metric]) / 2
            min_avg_idx = combined_avg.idxmin()
            min_avg_offset = train_data.loc[min_train_idx, 'offset'] 
            min_avg = combined_avg.min()
            
            # Add the results to the DataFrames
            train_mse_results.loc[len(train_mse_results)] = {'method': f"{subdir}-{min_train_offset}", 'mse': min_train_value}
            test_mse_results.loc[len(test_mse_results)] = {'method': f"{subdir}-{min_test_offset}", 'mse': min_test_value}
            min_avg_mse_results.loc[len(min_avg_mse_results)] = {'method': f"{subdir}-{min_avg_offset}", 'mse': min_avg}

train_mse_results.sort_values(by='mse').to_csv(os.path.join(output_dir, 'train_mse_results.csv'), index=False)
test_mse_results.sort_values(by='mse').to_csv(os.path.join(output_dir, 'test_mse_results.csv'), index=False)
min_avg_mse_results.sort_values(by='mse').to_csv(os.path.join(output_dir, 'min_avg_mse_results.csv'), index=False)

# Hitrate

output_dir = 'data/rf/hitrate'
os.makedirs(output_dir, exist_ok=True)

train_hitrate_results = pd.DataFrame(columns=['method', 'hitrate'])
test_hitrate_results = pd.DataFrame(columns=['method', 'hitrate'])
max_avg_hitrate_results = pd.DataFrame(columns=['method', 'hitrate'])

for subdir in os.listdir(input_dir):
    for target in os.listdir(os.path.join(input_dir, subdir)):
        rel_path = os.path.join(subdir, target)
        
        print(rel_path)
        
        metric = 'hitrate'
        
        train_data = pd.read_csv(os.path.join(input_dir, rel_path,f"train_{metric}_results.csv")).sort_values(by='offset')
        test_data = pd.read_csv(os.path.join(input_dir, rel_path, f"test_{metric}_results.csv")).sort_values(by='offset')

        max_train_idx = train_data[metric].idxmax()
        max_train_offset = train_data.loc[max_train_idx, 'offset'] 
        max_train_value = train_data.loc[max_train_idx, metric] 
        
        max_test_idx = test_data[metric].idxmax()
        max_test_offset = test_data.loc[max_test_idx, 'offset'] 
        max_test_value = test_data.loc[max_test_idx, metric] 

        combined_avg = (train_data[metric] + test_data[metric]) / 2
        max_avg = combined_avg.max()
        
        # Add the results to the DataFrames
        train_hitrate_results.loc[len(train_hitrate_results)] = {'method': subdir, 'hitrate': max_train_value}
        test_hitrate_results.loc[len(test_hitrate_results)] = {'method': subdir, 'hitrate': max_test_value}
        max_avg_hitrate_results.loc[len(max_avg_hitrate_results)] = {'method': subdir, 'hitrate': max_avg}

train_hitrate_results.sort_values(ascending=False, by='hitrate').to_csv(os.path.join(output_dir, 'train_hitrate_results.csv'), index=False)
test_hitrate_results.sort_values(ascending=False, by='hitrate').to_csv(os.path.join(output_dir, 'test_hitrate_results.csv'), index=False)
max_avg_hitrate_results.sort_values(ascending=False, by='hitrate').to_csv(os.path.join(output_dir, 'max_avg_hitrate_results.csv'), index=False)

