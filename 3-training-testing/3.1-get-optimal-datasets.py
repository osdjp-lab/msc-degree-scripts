#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np

mapping = {'d' : 'differenced',
           'dn' : 'diff-normalized',
           'ds' : 'diff-standardized',
           'ld' : 'log-differenced',
           'ldn' : 'log-diff-normalized',
           'lds' : 'log-diff-standardized',
           'ln' : 'log-normalized',
           'ls' : 'log-standardized',
           'l' : 'log-transformed',
           'n' : 'normalized',
           'r' : 'raw',
           's' : 'standardized'}

for model in ('mlp', 'svr', 'rf'):

    selection_dir = f"../data/2-window-selection/2-optimal/{model}/mse"
    
    mse_results = pd.read_csv(os.path.join(selection_dir, "min_avg_mse_results.csv"))
    min_idx = mse_results['mse'].idxmin()
    optimal_offset_dataset = mse_results.loc[min_idx, 'method']
    
    method, offset = optimal_offset_dataset.split('-')
    
    # ../.. due to relative path (used by scripts in subdirectories) 
    input_dir = f"../../data/1-preprocessing/6-split/{mapping[method]}/USD/{offset}"
    
    output_dir = f"../data/3-training-testing/{model}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "optimal_dataset_dir.txt"), 'w+') as file:
        file.write(input_dir)

