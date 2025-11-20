#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_dir = '../../data/rf'
metric = 'mse'

for subdir in os.listdir(input_dir):
    for target in os.listdir(os.path.join(input_dir, subdir)):
        rel_path = os.path.join(subdir, target)
        print(rel_path)

        train_data = pd.read_csv(os.path.join(input_dir, rel_path,f"train_{metric}_results.csv")).sort_values(by='offset')
        test_data = pd.read_csv(os.path.join(input_dir, rel_path, f"test_{metric}_results.csv")).sort_values(by='offset')
        
        textsize = 28
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_data['offset'].to_numpy(), train_data[metric].to_numpy(), label='Train')
        plt.plot(test_data['offset'].to_numpy(), test_data[metric].to_numpy(), label='Test')
        
        # Set title and labels
        # plt.title(f'{subdir} RF offset window {metric.upper()}', fontsize=textsize)
        plt.xlabel('Offset', fontsize=textsize)
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
        plt.xticks(fontsize=textsize)
        plt.yticks(fontsize=textsize)

        # Show the plot
        plt.show()

