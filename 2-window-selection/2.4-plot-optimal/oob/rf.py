#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/rf/oob'

min_oob = pd.read_csv(os.path.join(input_dir, "min_oob_results.csv")).sort_values(by='oob_error')

textsize = 28

print("min_oob")

plt.figure(figsize=(10, 6))

plt.bar(min_oob['method'].to_numpy(), min_oob['oob_error'].to_numpy(), label='Min-OOB')

# Set title and labels
# plt.title('RF minimum average MSE preprocessing methodology and offset', fontsize=textsize)
plt.xlabel('Preprocessing methodology and offset', fontsize=textsize)
plt.ylabel('OOB Error', fontsize=textsize)

# Add legend
plt.legend(fontsize=textsize)

# Enable scientific notation for y-axis
plt.ticklabel_format(axis='y',
                     style='scientific',
                     # useMathText=True,
                     scilimits=(0,0))

ax = plt.gca()
ax.set_yscale('log')
ax.yaxis.offsetText.set_fontsize(textsize)

# Increase tick label size
plt.xticks(fontsize=textsize)
plt.yticks(fontsize=textsize)

# Show the plot
plt.show()

