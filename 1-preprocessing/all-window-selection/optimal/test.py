#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'data/mlp/mse'

train_mse = pd.read_csv(os.path.join(input_dir, "train_mse_results.csv")).sort_values(by='mse')
test_mse = pd.read_csv(os.path.join(input_dir, "test_mse_results.csv")).sort_values(by='mse')
min_avg_mse = pd.read_csv(os.path.join(input_dir, "min_avg_mse_results.csv")).sort_values(by='mse')

textsize = 28

# plt.figure(figsize=(10, 6))
plt.figure()

plt.bar(train_mse['method'].to_numpy(), train_mse['mse'].to_numpy(), label='Train')

# Set title and labels
plt.title('MLP train set MSE per preprocessing methodology', fontsize=textsize)
plt.xlabel('Methodology', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

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
plt.xticks(fontsize=textsize, rotation=90)
plt.yticks(fontsize=textsize)

# Show the plot
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
plt.figure()

plt.bar(test_mse['method'].to_numpy(), test_mse['mse'].to_numpy(), label='Test')

# Set title and labels
plt.title('MLP test set MSE per preprocessing methodology', fontsize=textsize)
plt.xlabel('Methodology', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

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
plt.xticks(fontsize=textsize, rotation=90)
plt.yticks(fontsize=textsize)

# Show the plot
plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
plt.figure()

plt.bar(min_avg_mse['method'].to_numpy(), min_avg_mse['mse'].to_numpy(), label='Min-Avg')

# Set title and labels
plt.title('MLP average MSE per preprocessing methodology', fontsize=textsize)
plt.xlabel('Methodology', fontsize=textsize)
plt.ylabel('MSE', fontsize=textsize)

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
plt.xticks(fontsize=textsize, rotation=90)
plt.yticks(fontsize=textsize)

# Show the plot
plt.tight_layout()
plt.show()

