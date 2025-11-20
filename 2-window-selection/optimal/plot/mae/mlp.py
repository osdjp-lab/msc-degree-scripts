#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/mlp/mae'

train_mae = pd.read_csv(os.path.join(input_dir, "train_mae_results.csv")).sort_values(by='mae')
test_mae = pd.read_csv(os.path.join(input_dir, "test_mae_results.csv")).sort_values(by='mae')
min_avg_mae = pd.read_csv(os.path.join(input_dir, "min_avg_mae_results.csv")).sort_values(by='mae')

print("train_mae")

textsize = 28

plt.figure(figsize=(10, 6))

plt.bar(train_mae['method'].to_numpy(), train_mae['mae'].to_numpy(), label='Train')

# Set title and labels
# plt.title('MLP train set MAE preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing steps and offset', fontsize=textsize)
plt.ylabel('MAE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

# Enable scientific notation for y-axis
plt.ticklabel_format(axis='y',
                     style='scientific',
                     # useMathText=True,
                     scilimits=(0,0))

ax = plt.gca()
ax.yaxis.offsetText.set_fontsize(textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

print("test_mae")

plt.figure(figsize=(10, 6))

plt.bar(test_mae['method'].to_numpy(), test_mae['mae'].to_numpy(), label='Test')

# Set title and labels
# plt.title('MLP test set MAE preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing steps and offset', fontsize=textsize)
plt.ylabel('MAE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

# Enable scientific notation for y-axis
plt.ticklabel_format(axis='y',
                     style='scientific',
                     # useMathText=True,
                     scilimits=(0,0))

ax = plt.gca()
ax.yaxis.offsetText.set_fontsize(textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

print("min_avg_mae")

plt.figure(figsize=(10, 6))

plt.bar(min_avg_mae['method'].to_numpy(), min_avg_mae['mae'].to_numpy(), label='Min-Avg')

# Set title and labels
# plt.title('MLP minimum average MAE preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing steps and offset', fontsize=textsize)
plt.ylabel('MAE', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

# Enable scientific notation for y-axis
plt.ticklabel_format(axis='y',
                     style='scientific',
                     # useMathText=True,
                     scilimits=(0,0))

ax = plt.gca()
ax.yaxis.offsetText.set_fontsize(textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

