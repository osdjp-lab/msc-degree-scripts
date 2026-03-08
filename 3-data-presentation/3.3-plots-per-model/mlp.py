#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

model = 'mlp'

# MSE
dataset_type = 'log-normalized'
# Hitrate
# dataset_type = 'normalized'
# Other
# dataset_type = 'log-diff-normalized'
# dataset_type = 'diff-normalized'

offset = 1

plot_file= f"../../data/2-training-testing/{model}/{dataset_type}/41-USD/{offset}/test_pred_rev.csv"

data = pd.read_csv(plot_file)

date = data['Date']
date = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in date]

y_test = data['y_test']
y_pred = data['y_test_pred']

textsize = 28

plt.figure(figsize=(10, 6))

plt.plot(date, y_test.to_numpy(), label='Real')
plt.plot(date, y_pred.to_numpy(), label='Forecast')

# Set title and labels
plt.title(f'Real vs forecast (Offset {offset})', fontsize=textsize)
plt.xlabel('Date', fontsize=textsize)
plt.ylabel(f"USD", fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()
# plt.savefig(f"mlp-ln-{offset}.png", format='png', dpi=300, bbox_inches='tight')

