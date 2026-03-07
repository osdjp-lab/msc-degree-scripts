#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
model = 'rf'
dataset_type = 'log-transformed'
base_dir = Path("../../data/2-training-testing/optuna") / model / dataset_type / "41-USD"

output_dir = "./rf-ds-r"

os.makedirs(output_dir, exist_ok=True)

# Loop through offsets 1 to 12
for offset in range(1, 13):
    plot_file = base_dir / str(offset) / "test_pred_rev.csv"
    if not os.path.exists(plot_file):
        print(f"Warning: File not found for offset {offset}")
        continue
    
    data = pd.read_csv(plot_file)
    date_strs = data['Date']
    dates = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in date_strs]
    y_test = data['y_test']
    y_pred = data['y_test_pred']

    textsize = 14
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10,6))
    
    # Plot Real (thick black line)
    ax.plot(dates, y_test.to_numpy(), label='Real', linewidth=1, color='#1f77b4')
    
    # Plot Forecast (smooth lines only)
    ax.plot(dates, y_pred.to_numpy(), label='Forecast', linewidth=1, color='#ff7f0e')
    
    # Formatting
    ax.set_title(f'Real vs forecast (Offset {offset})', fontsize=textsize)
    ax.set_xlabel('Date', fontsize=textsize)
    ax.set_ylabel('USD', fontsize=textsize)
    ax.legend(fontsize=textsize)

    # Increase tick label size
    plt.xticks(fontsize=textsize)
    plt.yticks(fontsize=textsize)
    
    # Prevent x-axis label overlap
    #fig.autofmt_xdate()
    
    # Save with specific resolution
    plt.savefig(f'./{output_dir}/{offset}.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{offset}.png")

