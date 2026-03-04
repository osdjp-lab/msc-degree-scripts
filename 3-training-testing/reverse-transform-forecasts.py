#!/usr/bin/env python3

# Reverse transform the train and test set forecasts

import os
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# # # # # # # # # #

def rev_diff(initial, differenced, target):

    org, pred = differenced.columns[1:3]

    first_date = differenced['Date'].iloc[0]
    
    matching_indices = initial.index[initial['Date'] == first_date].tolist()
    if matching_indices:
        target_idx = matching_indices[0]
        prev_idx = target_idx - 1
        prev_date = initial.at[prev_idx, 'Date']
        prev_value = initial.at[prev_idx, target]
    
    first = {'Date': pd.Series(prev_date),
             org : pd.Series(prev_value),
             pred : pd.Series(prev_value)}
    first_df = pd.DataFrame(first)

    rev = pd.concat([first_df, differenced], axis=0)
    rev[org] = rev[org].cumsum()
    rev[pred] = rev[pred].cumsum()

    return rev

def rev_log(log_transformed):
    
    org, pred = log_transformed.columns[1:3]
    
    log_transformed[org] = np.exp(log_transformed[[org]])
    log_transformed[pred] = np.exp(log_transformed[[pred]])
    
    return log_transformed

def rev_norm(un_norm, norm, target):
    
    org, pred = norm.columns[1:3]
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(un_norm[[target]])
    
    norm[org] = scaler.inverse_transform(norm[[org]])
    norm[pred] = scaler.inverse_transform(norm[[pred]])
    
    return norm

def rev_std(un_std, std, target):
    
    org, pred = std.columns[1:3]
    
    scaler = StandardScaler()
    scaler.fit(un_std[[target]])
    
    std[org] = scaler.inverse_transform(std[[org]])
    std[pred] = scaler.inverse_transform(std[[pred]])
    
    return std

# # # # # # # # # #

INPUT_DIR = Path("../data/3-training-testing/optuna")

for model in os.listdir(INPUT_DIR):

    model_path = INPUT_DIR / model
    
    for dataset_type in os.listdir(model_path):
        if not os.path.isdir(model_path / dataset_type): 
            continue
        
        dataset_path = model_path / dataset_type
            
        for target in os.listdir(dataset_path):
            if not os.path.isdir(dataset_path / target): 
                continue

            target_path = dataset_path / target

            for offset in os.listdir(target_path):
                if not os.path.isdir(target_path / offset): 
                    continue

                offset_path = target_path / offset

                print(offset_path)

                test_set_forecast = pd.read_csv(offset_path / "test_pred.csv")
                train_set_forecast = pd.read_csv(offset_path / "train_pred.csv")

                if dataset_type == 'differenced':
                    ref_dataset = f'../data/1-preprocessing/3-groupings/raw/{target}.csv'

                    undifferenced_unsplit = pd.read_csv(ref_dataset)
                    
                    r_diff = rev_diff(undifferenced_unsplit, test_set_forecast, target)
                    r_diff.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_diff = rev_diff(undifferenced_unsplit, train_set_forecast, target)
                    r_diff.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'diff-normalized':
                    ref_dataset_1 = f'../data/1-preprocessing/3-groupings/raw/{target}.csv'
                    ref_dataset_2 = f'../data/1-preprocessing/3-groupings/differenced/{target}.csv'
                    
                    un_norm = pd.read_csv(ref_dataset_2)
                    undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                    
                    r_norm = rev_norm(un_norm, test_set_forecast, target)
                    r_norm_diff = rev_diff(undifferenced_unsplit, r_norm, target)
                    r_norm_diff.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_norm = rev_norm(un_norm, train_set_forecast, target)
                    r_norm_diff = rev_diff(undifferenced_unsplit, r_norm, target)
                    r_norm_diff.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'diff-standardized':
                    ref_dataset_1 = f'../data/1-preprocessing/3-groupings/raw/{target}.csv'
                    ref_dataset_2 = f'../data/1-preprocessing/3-groupings/differenced/{target}.csv'
                    
                    un_std = pd.read_csv(ref_dataset_2)
                    undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                    
                    r_std = rev_std(un_std, test_set_forecast, target)
                    r_std_diff = rev_diff(undifferenced_unsplit, r_std, target)
                    r_std_diff.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_std = rev_std(un_std, train_set_forecast, target)
                    r_std_diff = rev_diff(undifferenced_unsplit, r_std, target)
                    r_std_diff.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'log-differenced':
                    ref_dataset = f'../data/1-preprocessing/3-groupings/log-transformed/{target}.csv'
                    
                    undifferenced_unsplit = pd.read_csv(ref_dataset)

                    r_diff = rev_diff(undifferenced_unsplit, test_set_forecast, target)
                    r_diff_log = rev_log(r_diff)
                    r_diff_log.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_diff = rev_diff(undifferenced_unsplit, train_set_forecast, target)
                    r_diff_log = rev_log(r_diff)
                    r_diff_log.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'log-diff-normalized':
                    ref_dataset_1 = f'../data/1-preprocessing/3-groupings/log-transformed/{target}.csv'
                    ref_dataset_2 = f'../data/1-preprocessing/3-groupings/log-differenced/{target}.csv'
                    
                    un_norm = pd.read_csv(ref_dataset_2)
                    undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                    
                    r_norm = rev_norm(un_norm, test_set_forecast, target)
                    r_norm_diff = rev_diff(undifferenced_unsplit, r_norm, target)
                    r_norm_diff_log = rev_log(r_norm_diff)
                    r_norm_diff_log.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_norm = rev_norm(un_norm, train_set_forecast, target)
                    r_norm_diff = rev_diff(undifferenced_unsplit, r_norm, target)
                    r_norm_diff_log = rev_log(r_norm_diff)
                    r_norm_diff_log.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'log-diff-standardized':
                    ref_dataset_1 = f'../data/1-preprocessing/3-groupings/log-transformed/{target}.csv'
                    ref_dataset_2 = f'../data/1-preprocessing/3-groupings/log-differenced/{target}.csv'
                    
                    un_std = pd.read_csv(ref_dataset_2)
                    undifferenced_unsplit = pd.read_csv(ref_dataset_1)
                    
                    r_std = rev_std(un_std, test_set_forecast, target)
                    r_std_diff = rev_diff(undifferenced_unsplit, r_std, target)
                    r_std_diff_log = rev_log(r_std_diff)
                    r_std_diff_log.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_std = rev_std(un_std, train_set_forecast, target)
                    r_std_diff = rev_diff(undifferenced_unsplit, r_std, target)
                    r_std_diff_log = rev_log(r_std_diff)
                    r_std_diff_log.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'log-normalized':
                    ref_dataset = f'../data/1-preprocessing/3-groupings/log-transformed/{target}.csv'
                    
                    un_norm = pd.read_csv(ref_dataset)
                    
                    r_norm = rev_norm(un_norm, test_set_forecast, target)
                    r_norm_log = rev_log(r_norm)
                    r_norm_log.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_norm = rev_norm(un_norm, train_set_forecast, target)
                    r_norm_log = rev_log(r_norm)
                    r_norm_log.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'log-standardized':
                    ref_dataset = f'../data/1-preprocessing/3-groupings/log-transformed/{target}.csv'
                    
                    un_std = pd.read_csv(ref_dataset)
                    
                    r_std = rev_std(un_std, test_set_forecast, target)
                    r_std_log = rev_log(r_std)
                    r_std_log.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_std = rev_std(un_std, train_set_forecast, target)
                    r_std_log = rev_log(r_std)
                    r_std_log.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'log-transformed':
                    r_log = rev_log(test_set_forecast)
                    r_log.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_log = rev_log(train_set_forecast)
                    r_log.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'normalized':
                    ref_dataset = f'../data/1-preprocessing/3-groupings/raw/{target}.csv'

                    un_norm = pd.read_csv(ref_dataset)
                    
                    r_norm = rev_norm(un_norm, test_set_forecast, target)
                    r_norm.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_norm = rev_norm(un_norm, train_set_forecast, target)
                    r_norm.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'raw':
                    test_set_forecast.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    train_set_forecast.to_csv(offset_path / "train_pred_rev.csv", index=False)
                
                if dataset_type == 'standardized':
                    ref_dataset = f'../data/1-preprocessing/3-groupings/raw/{target}.csv'

                    un_std = pd.read_csv(ref_dataset)
                    
                    r_std = rev_std(un_std, test_set_forecast, target)
                    r_std.to_csv(offset_path / "test_pred_rev.csv", index=False)
                    
                    r_std = rev_std(un_std, train_set_forecast, target)
                    r_std.to_csv(offset_path / "train_pred_rev.csv", index=False)

