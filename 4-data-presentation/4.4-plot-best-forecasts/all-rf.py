#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

model = 'rf'
dataset_type = 'raw'
base_dir = Path("../../data/3-training-testing/optuna") / model / dataset_type / "41-USD"

# Load data for all offsets 1-12
all_data = {}
for offset in range(1, 13):
    plot_file = base_dir / str(offset) / "test_pred_rev.csv"
    if not plot_file.exists():
        print(f"Warning: File not found for offset {offset}: {plot_file}")
        continue
    data = pd.read_csv(plot_file)
    date_strs = data['Date']
    dates = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in date_strs]
    all_data[offset] = {'dates': dates, 'y_test': data['y_test'], 'y_test_pred': data['y_test_pred']}

if not all_data:
    raise ValueError("No data files found for any offset!")

# Use first dataset's dates and y_test (assuming identical across offsets)
dates = all_data[1]['dates']
y_test = all_data[1]['y_test']

fig, ax = plt.subplots(figsize=(14, 8))

# Plot y_test once, prominent (thick black line with markers)
ax.plot(dates, y_test.to_numpy(), 'k-', linewidth=3, label='Real')

# Plot each y_test_pred with auto-color cycling, smooth lines only (no markers, no dashes)
for offset in range(1, 13):
    if offset not in all_data:
        continue
    data = all_data[offset]
    ax.plot(data['dates'], data['y_test_pred'].to_numpy(), label=f'Offset {offset}', linewidth=2)

# Formatting
ax.set_title('Real vs Forecasts (Offsets 1-12)', fontsize=28)
ax.set_xlabel('Date', fontsize=28)
ax.set_ylabel("USD", fontsize=28)
ax.legend(loc='upper left', fontsize=20)
ax.tick_params(axis='both', labelsize=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

