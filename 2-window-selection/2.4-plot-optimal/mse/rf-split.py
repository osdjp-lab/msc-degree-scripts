#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/rf/mse'

train_mse = pd.read_csv(os.path.join(input_dir, "train_mse_results.csv")).sort_values(by='mse')
test_mse = pd.read_csv(os.path.join(input_dir, "test_mse_results.csv")).sort_values(by='mse')
min_avg_mse = pd.read_csv(os.path.join(input_dir, "min_avg_mse_results.csv")).sort_values(by='mse')

textsize = 28

datasets = [
    {"name": "train_mse", "data": train_mse},
    {"name": "test_mse", "data": test_mse},
    {"name": "min_avg_mse", "data": min_avg_mse},
]

for dataset in datasets:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    first_six = dataset["data"].head(6)
    remaining_six = dataset["data"].tail(6)

    axs[0].bar(first_six['method'].to_numpy(), first_six['mse'].to_numpy(), label='First 6')
    axs[0].set_xlabel('Preprocessing methodology and offset', fontsize=textsize)
    axs[0].set_ylabel('MSE', fontsize=textsize)
    axs[0].legend(fontsize=textsize)
    axs[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    axs[0].yaxis.offsetText.set_fontsize(textsize)
    axs[0].tick_params(axis='x', labelsize=textsize)
    axs[0].tick_params(axis='y', labelsize=textsize)

    axs[1].bar(remaining_six['method'].to_numpy(), remaining_six['mse'].to_numpy(), label='Remaining 6')
    axs[1].set_xlabel('Preprocessing methodology and offset', fontsize=textsize)
    axs[1].set_ylabel('MSE', fontsize=textsize)
    axs[1].legend(fontsize=textsize)
    axs[1].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    axs[1].yaxis.offsetText.set_fontsize(textsize)
    axs[1].tick_params(axis='x', labelsize=textsize)
    axs[1].tick_params(axis='y', labelsize=textsize)

    fig.tight_layout()
    plt.suptitle(dataset["name"], fontsize=textsize)
    plt.show()

