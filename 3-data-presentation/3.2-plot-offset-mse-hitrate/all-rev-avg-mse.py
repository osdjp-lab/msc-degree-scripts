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

INPUT_DIR_MLP = Path("../../data/2-training-testing/mlp/log-normalized/41-USD/")
mse_mlp = pd.read_csv(INPUT_DIR_MLP / 'rev_avg_mse.csv').sort_values(by='offset')

######
# RF
######

INPUT_DIR_RF = Path("../../data/2-training-testing/rf/raw/41-USD/")
mse_rf = pd.read_csv(INPUT_DIR_RF / 'rev_avg_mse.csv').sort_values(by='offset')

######
# SVR
######

INPUT_DIR_SVR = Path("../../data/2-training-testing/svr/log-diff-standardized/41-USD/")
mse_svr = pd.read_csv(INPUT_DIR_SVR / 'rev_avg_mse.csv').sort_values(by='offset')

########
# ARIMA
########

INPUT_DIR_SVR = Path("../../data/2-training-testing/arima/log-differenced/41-USD/")
mse_arima = pd.read_csv(INPUT_DIR_SVR / 'rev_avg_mse.csv').sort_values(by='offset')

#####
# RW
#####

INPUT_DIR_SVR = Path("../../data/2-training-testing/rw/raw/41-USD/")
mse_rw = pd.read_csv(INPUT_DIR_SVR / 'rev_avg_mse.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

# Offsets from any model (they should match across models)
x = mse_mlp['offset'].to_numpy()
width = 0.18  # Width for grouped bars

# Grouped bar chart
plt.bar(x - 2*width, mse_mlp['avg_mse'].to_numpy(), width, label="MLP-ln")
plt.bar(x - width, mse_rf['avg_mse'].to_numpy(), width, label="RF-r")
plt.bar(x, mse_svr['avg_mse'].to_numpy(), width, label="SVR-lds")
plt.bar(x + width, mse_arima['avg_mse'].to_numpy(), width, label="ARIMA-ld")
plt.bar(x + 2*width, mse_rw['avg_mse'].to_numpy(), width, label="RW-r")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(x)

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

