#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'data'

common_name = 'features'

# Plot forecast

data = pd.read_csv(os.path.join(input_dir, f'{common_name}_forecast.csv'))

date = data['Date']
date = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in date]

y_test = data['y_test']
y_pred = data['y_pred']

textsize = 28

plt.figure(figsize=(10, 6))

plt.plot(date, y_test.to_numpy(), label='Real')
plt.plot(date, y_pred.to_numpy(), label='Forecast')

# Set title and labels
plt.title(f'Real vs forecast', fontsize=textsize)
plt.xlabel('Date', fontsize=textsize)
plt.ylabel('USD', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

# Plot fit results

data = pd.read_csv(os.path.join(input_dir, f'{common_name}_fit.csv'))

max_features = data['max_features']
negmse = data['negmse']

textsize = 28

plt.figure(figsize=(10, 6))

plt.plot(max_features.to_numpy(), negmse.to_numpy(), label='Neg-MSE')

# Set title and labels
# plt.title(f'Number of hidden layer nodes vs NMSE', fontsize=textsize)
plt.xlabel('Number of features', fontsize=textsize)
plt.ylabel('Negative Mean Squared Error', fontsize=textsize)

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


