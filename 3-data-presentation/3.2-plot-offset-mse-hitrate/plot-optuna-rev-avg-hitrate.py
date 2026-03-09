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

INPUT_DIR = Path("../../data/2-training-testing/mlp/log-diff-normalized/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="MLP-ldn")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(hitrate['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# plt.tight_layout()
plt.show()

#####
# RF
#####

INPUT_DIR = Path("../../data/2-training-testing/rf/log-transformed/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="RF-l")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(hitrate['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# plt.tight_layout()
plt.show()

######
# SVR
######

INPUT_DIR = Path("../../data/2-training-testing/svr/log-diff-standardized/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="SVR-lds")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(hitrate['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# plt.tight_layout()
plt.show()

########
# ARIMA
########

INPUT_DIR = Path("../../data/2-training-testing/arima/diff-standardized/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="ARIMA-ds")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(hitrate['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# plt.tight_layout()
plt.show()

#####
# RW
#####

INPUT_DIR = Path("../../data/2-training-testing/rw/raw/41-USD/")

hitrate = pd.read_csv(INPUT_DIR / 'rev_avg_hitrate.csv').sort_values(by='offset')

textsize = 28

plt.figure(figsize=(12, 6))

plt.bar(hitrate['offset'].to_numpy(), hitrate['avg_hitrate'].to_numpy(), label="RW-r")

plt.xlabel('Offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

plt.xticks(hitrate['offset'].to_numpy())

# Increase tick label size
plt.xticks(fontsize=textsize-4)
plt.yticks(fontsize=textsize)

# plt.tight_layout()
plt.show()


