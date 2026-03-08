#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
mlp_mse = Path("../../data/2-training-testing/mlp/rev_mse_results.csv")
rf_mse = Path("../../data/2-training-testing/rf/rev_mse_results.csv")
svr_mse = Path("../../data/2-training-testing/svr/rev_mse_results.csv")

# Read all CSV files and find minimum avg_mse for each model
mlp_data = pd.read_csv(mlp_mse)
rf_data = pd.read_csv(rf_mse)
svr_data = pd.read_csv(svr_mse)

# Get the lowest avg_mse value from each file
mlp_min = mlp_data['avg_mse'].min()
rf_min = rf_data['avg_mse'].min()
svr_min = svr_data['avg_mse'].min()

# Data for 3-bar chart
models = ['MLP', 'RF', 'SVR']
min_values = [mlp_min, rf_min, svr_min]

textsize = 24

# Create 3-bar chart
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(models, min_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.6)

# Customize
ax.set_xlabel('Model', fontsize=textsize)
ax.set_ylabel('Lowest Avg MSE', fontsize=textsize)
#ax.set_title('Lowest MSE Achieved by Each Model', fontsize=16)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}',
            ha='center', va='bottom', fontsize=textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

plt.tight_layout()
#plt.savefig('lowest_mse_by_model.png', dpi=300, bbox_inches='tight')
plt.show()

