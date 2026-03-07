#!/usr/bin/env python3

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

######
# MLP
######

INPUT_DIR = Path("../../data/3-training-testing/optuna/mlp/log-normalized/41-USD/")

mse = pd.read_csv(INPUT_DIR / 'rev_avg_mse.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(mse['offset'].to_numpy(), mse['avg_mse'].to_numpy(), label="MLP-ln")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(mse['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

#####
# RF
#####

INPUT_DIR = Path("../../data/3-training-testing/optuna/rf/raw/41-USD/")

mse = pd.read_csv(INPUT_DIR / 'rev_avg_mse.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(mse['offset'].to_numpy(), mse['avg_mse'].to_numpy(), label="RF-r")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(mse['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

######
# SVR
######

INPUT_DIR = Path("../../data/3-training-testing/optuna/svr/log-diff-standardized/41-USD/")

mse = pd.read_csv(INPUT_DIR / 'rev_avg_mse.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(mse['offset'].to_numpy(), mse['avg_mse'].to_numpy(), label="SVR-lds")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(mse['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

