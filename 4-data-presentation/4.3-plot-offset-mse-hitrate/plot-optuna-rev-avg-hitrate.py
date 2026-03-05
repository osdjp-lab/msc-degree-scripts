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

INPUT_DIR = Path("../../data/3-training-testing/optuna/mlp/normalized/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(10, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="MLP Normalized")
#plt.plot(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="MLP Log-Diff-Normalized")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.ylim(0.4, 0.6)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

#####
# RF
#####

INPUT_DIR = Path("../../data/3-training-testing/optuna/rf/log-transformed/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(10, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="RF Log-Transformed")
#plt.plot(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="RF Log-Diff-Normalized")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.ylim(0.4, 0.8)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

######
# SVR
######

INPUT_DIR = Path("../../data/3-training-testing/optuna/svr/log-diff-standardized/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(10, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="SVR Log-Diff-Standardized")
#plt.plot(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="SVR Diff-Standardized")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.ylim(0.5, 0.9)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

