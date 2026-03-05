#!/usr/bin/env python3

"""Caluculate the average mse, mae and hitrate for each model,
dataset type, target and offset"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_DIR= Path("../data/3-training-testing/optuna")

for model in os.listdir(INPUT_DIR):
    if not os.path.isdir(INPUT_DIR / model): 
        continue
    
    model_path = INPUT_DIR / model

    mse_results = {}
    mae_results = {}
    hitrate_results = {}
    
    for dataset_type in os.listdir(model_path):
        if not os.path.isdir(model_path / dataset_type): 
            continue
        
        dataset_path = model_path / dataset_type
            
        for target in os.listdir(dataset_path):
            if not os.path.isdir(dataset_path / target): 
                continue

            target_path = dataset_path / target

            print(target_path)

            for metric in ('mse','mae','hitrate'):
        
                print(metric)

                # Load data
                df_train = pd.read_csv(target_path / f"train_{metric}_results.csv")
                df_test = pd.read_csv(target_path / f"test_{metric}_results.csv")
                
                # Merge on offset (handles out-of-order rows)
                df_merged = pd.merge(df_train, df_test, on='offset', suffixes=('_train', '_test'))
                
                # Compute average
                df_avg = pd.DataFrame({
                    'offset': df_merged['offset'],
                    f"avg_{metric}": (df_merged[f"{metric}_train"] + df_merged[f"{metric}_test"]) / 2
                })

                print(target_path / f"avg_{metric}.csv")
                
                # Save sorted by offset
                df_avg = df_avg.sort_values('offset').reset_index(drop=True)
                df_avg.to_csv(target_path / f"avg_{metric}.csv", index=False)

        mse_results[dataset_type] = pd.read_csv(dataset_path / '41-USD' / 'avg_mse.csv')['avg_mse'].mean()
        
        mae_results[dataset_type] = pd.read_csv(dataset_path / '41-USD' / 'avg_mae.csv')['avg_mae'].mean()
        
        hitrate_results[dataset_type] = pd.read_csv(dataset_path / '41-USD' / 'avg_hitrate.csv')['avg_hitrate'].mean()

    df = pd.DataFrame(list(mse_results.items()), columns=['type', 'avg_mse'])
    df.to_csv(model_path / 'mse_results.csv', index=False)
    
    df = pd.DataFrame(list(mae_results.items()), columns=['type', 'avg_mae'])
    df.to_csv(model_path / 'mae_results.csv', index=False)
    
    df = pd.DataFrame(list(hitrate_results.items()), columns=['type', 'avg_hitrate'])
    df.to_csv(model_path / 'hitrate_results.csv', index=False)

