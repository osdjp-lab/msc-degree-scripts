#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = 'data'

common_name = 'all'

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

