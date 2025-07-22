#!/usr/bin/env python3

# Rename dataset testing results to abbreviated forms
# for better graphical display

import os

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

data_dir = './data'

for subdir in os.listdir(data_dir):
    for target in os.listdir(os.path.join(data_dir, subdir)):
        print("----------------")
        print(os.path.join(data_dir, subdir, target))
        print(os.path.join(data_dir, subdir, mapping[target]))
        os.rename(os.path.join(data_dir, subdir, target),
                  os.path.join(data_dir, subdir, mapping[target]))

