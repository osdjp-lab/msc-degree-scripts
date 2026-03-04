#!/usr/bin/env python3

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("../data/3-training-testing/optuna")

mapping = {'differenced' : 'd',
           'diff-normalized' : 'dn',
           'diff-standardized' : 'ds',
           'log-differenced' : 'ld',
           'log-diff-normalized' : 'ldn',
           'log-diff-standardized' : 'lds',
           'log-normalized' : 'ln',
           'log-standardized' : 'ls',
           'log-transformed' : 'l',
           'normalized' : 'n',
           'raw' : 'r',
           'standardized' : 's'}

for model in os.listdir(INPUT_DIR):

    mse = pd.read_csv(INPUT_DIR / model / 'rev_mse_results.csv').sort_values(by='avg_mse')

    mse = mse.drop(mse[mse['type'] == 'raw'].index)

    mse['type'] = mse['type'].map(mapping)

    textsize = 28

    plt.figure(figsize=(10, 6))

    plt.bar(mse['type'].to_numpy(), mse['avg_mse'].to_numpy(), label=str(model).upper())

    plt.xlabel('Preprocessing steps', fontsize=textsize)
    plt.ylabel('MSE', fontsize=textsize)

    # Add legend
    plt.legend(fontsize=textsize)

    # Enable scientific notation for y-axis
    # plt.ticklabel_format(axis='y',
    #                      style='scientific',
    #                      # useMathText=True,
    #                      scilimits=(0,0))

    # ax = plt.gca()
    # ax.yaxis.offsetText.set_fontsize(textsize)

    # Increase tick label size
    plt.xticks(fontsize=textsize)
    plt.yticks(fontsize=textsize)

    # Show the plot
    plt.show()

