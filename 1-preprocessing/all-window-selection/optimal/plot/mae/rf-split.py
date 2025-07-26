#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/rf/mae'

train_mae = pd.read_csv(os.path.join(input_dir, "train_mae_results.csv")).sort_values(by='mae')
test_mae = pd.read_csv(os.path.join(input_dir, "test_mae_results.csv")).sort_values(by='mae')
min_avg_mae = pd.read_csv(os.path.join(input_dir, "min_avg_mae_results.csv")).sort_values(by='mae')

textsize = 28

datasets = [
    {"name": "train_mae", "data": train_mae},
    {"name": "test_mae", "data": test_mae},
    {"name": "min_avg_mae", "data": min_avg_mae},
]

for dataset in datasets:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    first_six = dataset["data"].head(6)
    remaining_six = dataset["data"].tail(6)

    axs[0].bar(first_six['method'].to_numpy(), first_six['mae'].to_numpy(), label='First 6')
    axs[0].set_xlabel('Preprocessing methodology and offset', fontsize=textsize)
    axs[0].set_ylabel('MAE', fontsize=textsize)
    axs[0].legend(fontsize=textsize)
    axs[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    axs[0].yaxis.offsetText.set_fontsize(textsize)
    axs[0].tick_params(axis='x', labelsize=textsize)
    axs[0].tick_params(axis='y', labelsize=textsize)

    axs[1].bar(remaining_six['method'].to_numpy(), remaining_six['mae'].to_numpy(), label='Remaining 6')
    axs[1].set_xlabel('Preprocessing methodology and offset', fontsize=textsize)
    axs[1].set_ylabel('MAE', fontsize=textsize)
    axs[1].legend(fontsize=textsize)
    axs[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    axs[1].yaxis.offsetText.set_fontsize(textsize)
    axs[1].tick_params(axis='x', labelsize=textsize)
    axs[1].tick_params(axis='y', labelsize=textsize)

    fig.tight_layout()
    plt.suptitle(dataset["name"], fontsize=textsize)
    plt.show()

