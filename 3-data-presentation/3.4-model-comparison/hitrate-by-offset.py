#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MLP_DIR = Path("../../data/2-training-testing/mlp/log-diff-normalized/41-USD/")
RF_DIR = Path("../../data/2-training-testing/rf/log-transformed/41-USD/")
SVR_DIR = Path("../../data/2-training-testing/svr/log-diff-standardized/41-USD/")

OUTPUT_DIR = Path("./forecast-comparison/hitrate")
os.makedirs(OUTPUT_DIR, exist_ok=True)

textsize = 14

for offset in range(1, 13):
    print(f"Creating plot for offset {offset}...")
    
    # Load data from each model
    mlp_file = MLP_DIR / str(offset) / 'test_pred_rev.csv'
    rf_file = RF_DIR / str(offset) / 'test_pred_rev.csv' 
    svr_file = SVR_DIR / str(offset) / 'test_pred_rev.csv'
    
    # Read CSVs
    df_mlp = pd.read_csv(mlp_file)
    df_rf = pd.read_csv(rf_file)
    df_svr = pd.read_csv(svr_file)
    
    # Use first file's y_test as Real (identical across files)
    df = pd.DataFrame({
        'Date': df_mlp['Date'],
        'Real': df_mlp['y_test'],
        'MLP': df_mlp['y_test_pred'],
        'RF': df_rf['y_test_pred'],
        'SVR': df_svr['y_test_pred']
    })
    
    # Convert Date to datetime for plotting
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['Date'], df['Real'], label='Real', color='black')
    plt.plot(df['Date'], df['MLP'], label='MLP', color='#1f77b4', alpha=0.8)
    plt.plot(df['Date'], df['RF'], label='RF', color='#ff7f0e', alpha=0.8)
    plt.plot(df['Date'], df['SVR'], label='SVR', color='#2ca02c', alpha=0.8)
    
    plt.xlabel('Date', fontsize=textsize)
    plt.ylabel('USD', fontsize=textsize)
    plt.title(f'Forecast Comparison - Offset {offset}', fontsize=textsize, pad=20)
    plt.legend(fontsize=textsize)
    plt.grid(True, alpha=0.3)
    
    # Rotate date labels for readability
    plt.xticks(fontsize=textsize)
    # plt.xticks(fontsize=textsize, rotation=45)
    plt.yticks(fontsize=textsize)
    # plt.tight_layout()
    
    # Save plot
    plt.savefig(OUTPUT_DIR / f'{offset}.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close()

print("All 12 plots created!")

