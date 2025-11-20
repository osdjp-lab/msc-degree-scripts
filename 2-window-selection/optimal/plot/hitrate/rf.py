#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/rf/hitrate'

train_hitrate = pd.read_csv(os.path.join(input_dir, "train_hitrate_results.csv")).sort_values(by='hitrate')
test_hitrate = pd.read_csv(os.path.join(input_dir, "test_hitrate_results.csv")).sort_values(by='hitrate')
max_avg_hitrate = pd.read_csv(os.path.join(input_dir, "max_avg_hitrate_results.csv")).sort_values(by='hitrate')

textsize = 20

print("train_hitrate")

plt.figure(figsize=(10, 6))

plt.bar(train_hitrate['method'].to_numpy(), train_hitrate['hitrate'].to_numpy(), label='Train')

# Set title and labels
# plt.title('RF train set hitrate preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing methodology and offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

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

print("test_hitrate")

plt.figure(figsize=(10, 6))

plt.bar(test_hitrate['method'].to_numpy(), test_hitrate['hitrate'].to_numpy(), label='Test')

# Set title and labels
# plt.title('RF test set hitrate preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing methodology and offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

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

print("max_avg_hitrate")

plt.figure(figsize=(10, 6))

plt.bar(max_avg_hitrate['method'].to_numpy(), max_avg_hitrate['hitrate'].to_numpy(), label='Max-Avg')

# Set title and labels
# plt.title('RF maximum average hitrate preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing methodology and offset', fontsize=textsize)
plt.ylabel('Hitrate', fontsize=textsize)

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

