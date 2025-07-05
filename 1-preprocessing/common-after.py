#!/usr/bin/env python3

# Split data into train-test sets sequentially

from preprocessing import split_data

nr_lags = 500

# MLP

input_dir='data/nn/7-svr-selection'
output_dir='data/nn/8-split/'
split_data(input_dir, output_dir, nr_lags)

# SVR

input_dir='data/svr/7-svr-selection'
output_dir='data/svr/8-split/'
split_data(input_dir, output_dir, nr_lags)

# RF

input_dir='data/rf/7-svr-selection'
output_dir='data/rf/8-split/'
split_data(input_dir, output_dir, nr_lags)

