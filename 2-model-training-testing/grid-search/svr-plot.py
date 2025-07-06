#!/usr/bin/env python3

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot

input_dir = 'svr-forecasts'

for input_file in [f for f in os.listdir(input_dir) if f.endswith('.csv')]:

    scoring = os.path.splitext(input_file)[0]

    data = pd.read_csv(os.path.join(input_dir, input_file))
    
    date = data['Date']
    date = [datetime.datetime.strptime(elem, '%Y-%m-%d') for elem in date]
    
    y_test = data['y_test']
    y_pred = data['y_pred']
    
    textsize = 28
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(date, y_test.to_numpy(), label='Real')
    plt.plot(date, y_pred.to_numpy(), label='Forecast')
    
    # Set title and labels
    plt.title(f'Real vs {scoring} forecast', fontsize=textsize)
    plt.xlabel('Date', fontsize=textsize)
    plt.ylabel('USD', fontsize=textsize)
    
    # Add legend
    plt.legend(fontsize=textsize)
    
    # Set y-axis limits to ensure both data ranges are visible
    # plt.ylim([-2,1])
    
    # Increase tick label size
    plt.xticks(fontsize=textsize)
    plt.yticks(fontsize=textsize)
    
    # Show the plot
    plt.show()
    
