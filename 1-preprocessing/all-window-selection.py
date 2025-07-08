#!/usr/bin/env python3

# Window size selection for all dataset preprocessing variations

from preprocessing import split_data_alt

features_dir = 'data/decorrelation-tests/4-features'
splits_dir = 'data/decorrelation-tests/5-split'

split_data_alt(features_dir, splits_dir)

