#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/rf/oob'

min_oob = pd.read_csv(os.path.join(input_dir, "min_oob_results.csv")).sort_values(by='oob_error')

textsize = 20

print("min_oob")

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Split the data into two parts
first_six = min_oob.head(6)
remaining_six = min_oob.tail(6)

# Plot the first six values
axs[0].bar(first_six['method'].to_numpy(), first_six['oob_error'].to_numpy(), label='Min-OOB')
axs[0].set_xlabel('Preprocessing methodology and offset', fontsize=textsize)
axs[0].set_ylabel('OOB Error', fontsize=textsize)
axs[0].legend(fontsize=textsize)
axs[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
axs[0].yaxis.offsetText.set_fontsize(textsize)
axs[0].tick_params(axis='x', labelsize=textsize)
axs[0].tick_params(axis='y', labelsize=textsize)

# Plot the remaining six values
axs[1].bar(remaining_six['method'].to_numpy(), remaining_six['oob_error'].to_numpy(), label='Min-OOB')
axs[1].set_xlabel('Preprocessing methodology and offset', fontsize=textsize)
axs[1].set_ylabel('OOB Error', fontsize=textsize)
axs[1].legend(fontsize=textsize)
axs[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
axs[1].yaxis.offsetText.set_fontsize(textsize)
axs[1].tick_params(axis='x', labelsize=textsize)
axs[1].tick_params(axis='y', labelsize=textsize)

# Layout so plots do not overlap
fig.tight_layout()

# Show the plot
plt.show()

