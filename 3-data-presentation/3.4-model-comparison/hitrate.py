#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
mlp_hitrate = Path("../../data/2-training-testing/mlp/rev_hitrate_results.csv")
rf_hitrate = Path("../../data/2-training-testing/rf/rev_hitrate_results.csv")
svr_hitrate = Path("../../data/2-training-testing/svr/rev_hitrate_results.csv")

# Read all CSV files and find maximum avg_hitrate for each model
mlp_data = pd.read_csv(mlp_hitrate)
rf_data = pd.read_csv(rf_hitrate)
svr_data = pd.read_csv(svr_hitrate)

# Get the highest avg_hitrate value from each file
mlp_max = mlp_data['avg_hitrate'].max()
rf_max = rf_data['avg_hitrate'].max()
svr_max = svr_data['avg_hitrate'].max()

# Data for 3-bar chart
models = ['MLP', 'RF', 'SVR']
max_values = [mlp_max, rf_max, svr_max]

textsize = 24

# Create 3-bar chart
fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.bar(models, max_values, color=['#1f77b4', '#ff7f0e', '#2ca02c'], width=0.6)

# Customize
ax.set_xlabel('Model', fontsize=textsize)
ax.set_ylabel('Largest Avg Hitrate', fontsize=textsize)
#ax.set_title('Highest Hitrate Achieved by Each Model', fontsize=16)

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
#plt.savefig('highest_hitrate_by_model.png', dpi=300, bbox_inches='tight')
plt.show()

