#!/usr/bin/env python3

import os
from preprocessing import *

input_dir = '../data/0-raw'
output_dir = '../data/1-preprocessing'
os.makedirs(output_dir, exist_ok=True)

# Common handling of missing values

missing_dir = os.path.join(output_dir, '0-missing-values')
os.makedirs(missing_dir, exist_ok=True)

remove_variables_with_missing_values(
        os.path.join(input_dir, 'eurofxref.csv'),
        os.path.join(missing_dir, 'eurofxref-no-nan.csv'))

interpolate(
        os.path.join(missing_dir, 'eurofxref-no-nan.csv'),
        os.path.join(missing_dir, 'eurofxref-interpolated.csv'))

remove_variables_with_missing_values(
        os.path.join(input_dir, 'real-cpi-deflated-eer-abbr.csv'),
        os.path.join(missing_dir, 'real-cpi-deflated-eer-abbr-no-nan.csv'))

# Common creation of merged dataset

merged_dir = os.path.join(output_dir, '1-merged')
os.makedirs(merged_dir, exist_ok=True)

merge_datasets(
        os.path.join(merged_dir, 'merged_dataset.csv'),
        os.path.join(missing_dir, 'eurofxref-interpolated.csv'),
        os.path.join(missing_dir, 'real-cpi-deflated-eer-abbr-no-nan.csv'))

# All combinations of preprocessing steps

base_file = os.path.join(merged_dir, 'merged_dataset.csv')

correlated_dir = os.path.join(output_dir, '2-correlated')
os.makedirs(correlated_dir, exist_ok=True)

# Raw

log_transform(base_file,
              os.path.join(correlated_dir, 'log-transformed.csv'))

difference(base_file,
           os.path.join(correlated_dir, 'differenced.csv'))

normalize(base_file,
          os.path.join(correlated_dir, 'normalized.csv'),
          (-1,1))

standardize(base_file,
            os.path.join(correlated_dir, 'standardized.csv'))

# 1st degree combinations

# Logs

difference(os.path.join(correlated_dir, 'log-transformed.csv'),
           os.path.join(correlated_dir, 'log-differenced.csv'))

normalize(os.path.join(correlated_dir, 'log-transformed.csv'),
          os.path.join(correlated_dir, 'log-normalized.csv'))

standardize(os.path.join(correlated_dir, 'log-transformed.csv'),
            os.path.join(correlated_dir, 'log-standardized.csv'))

# Diffs

normalize(os.path.join(correlated_dir, 'differenced.csv'),
          os.path.join(correlated_dir, 'diff-normalized.csv'))

standardize(os.path.join(correlated_dir, 'differenced.csv'),
            os.path.join(correlated_dir, 'diff-standardized.csv'))

# 2nd degree combinations

normalize(os.path.join(correlated_dir, 'log-differenced.csv'),
          os.path.join(correlated_dir, 'log-diff-normalized.csv'))

standardize(os.path.join(correlated_dir, 'log-differenced.csv'),
            os.path.join(correlated_dir, 'log-diff-standardized.csv'))

############################################

print('Creating groupings')

groupings = {'USD': ['JPY','CZK','DKK','GBP','HUF','PLN','SEK','CHF','NOK','AUD','CAD','HKD','KRW','NZD','SGD','ZAR']}

groupings_dir = os.path.join(output_dir, '3-groupings')
os.makedirs(groupings_dir, exist_ok=True)

create_groupings(base_file,
                 os.path.join(groupings_dir, 'raw'), groupings)

create_groupings(os.path.join(correlated_dir, 'log-transformed.csv'),
                 os.path.join(groupings_dir, 'log-transformed'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'differenced.csv'),
                 os.path.join(groupings_dir, 'differenced'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'normalized.csv'),
                 os.path.join(groupings_dir, 'normalized'), groupings)

create_groupings(os.path.join(correlated_dir, 'standardized.csv'),
                 os.path.join(groupings_dir, 'standardized'), groupings)

# Combinations

create_groupings(os.path.join(correlated_dir, 'log-differenced.csv'),
                 os.path.join(groupings_dir, 'log-differenced'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'log-normalized.csv'),
                 os.path.join(groupings_dir, 'log-normalized'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'log-standardized.csv'),
                 os.path.join(groupings_dir, 'log-standardized'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'diff-normalized.csv'),
                 os.path.join(groupings_dir, 'diff-normalized'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'diff-standardized.csv'),
                 os.path.join(groupings_dir, 'diff-standardized'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'log-diff-normalized.csv'),
                 os.path.join(groupings_dir, 'log-diff-normalized'),
                 groupings)

create_groupings(os.path.join(correlated_dir, 'log-diff-standardized.csv'),
                 os.path.join(groupings_dir, 'log-diff-standardized'),
                 groupings)

############################################

print('Decorrelating variables')

decorrelated_dir = os.path.join(output_dir, '4-decorrelated')
os.makedirs(decorrelated_dir, exist_ok=True)

decorrelate(
            os.path.join(groupings_dir, 'raw'),
            os.path.join(decorrelated_dir, 'raw')
            )

decorrelate(
            os.path.join(groupings_dir, 'log-transformed'),
            os.path.join(decorrelated_dir, 'log-transformed')
            )

decorrelate(
            os.path.join(groupings_dir, 'differenced'),
            os.path.join(decorrelated_dir, 'differenced')
            )

decorrelate(
            os.path.join(groupings_dir, 'normalized'),
            os.path.join(decorrelated_dir, 'normalized')
            )

decorrelate(
            os.path.join(groupings_dir, 'standardized'),
            os.path.join(decorrelated_dir, 'standardized')
            )

# Combinations

decorrelate(
            os.path.join(groupings_dir, 'log-differenced'),
            os.path.join(decorrelated_dir, 'log-differenced')
            )

decorrelate(
            os.path.join(groupings_dir, 'log-normalized'),
            os.path.join(decorrelated_dir, 'log-normalized')
            )

decorrelate(
            os.path.join(groupings_dir, 'log-standardized'),
            os.path.join(decorrelated_dir, 'log-standardized')
            )

decorrelate(
            os.path.join(groupings_dir, 'diff-normalized'),
            os.path.join(decorrelated_dir, 'diff-normalized')
            )

decorrelate(
            os.path.join(groupings_dir, 'diff-standardized'),
            os.path.join(decorrelated_dir, 'diff-standardized')
            )

decorrelate(
            os.path.join(groupings_dir, 'log-diff-normalized'),
            os.path.join(decorrelated_dir, 'log-diff-normalized')
            )

decorrelate(
            os.path.join(groupings_dir, 'log-diff-standardized'),
            os.path.join(decorrelated_dir, 'log-diff-standardized')
            )

############################################

# Get time lags

features_dir = os.path.join(output_dir, '5-features')
os.makedirs(features_dir, exist_ok=True)

nr_lags = 50

create_features_alt(
            os.path.join(decorrelated_dir, 'raw'),
            os.path.join(features_dir, 'raw'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'log-transformed'),
            os.path.join(features_dir, 'log-transformed'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'differenced'),
            os.path.join(features_dir, 'differenced'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'normalized'),
            os.path.join(features_dir, 'normalized'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'standardized'),
            os.path.join(features_dir, 'standardized'),
            nr_lags
            )

# Combinations

create_features_alt(
            os.path.join(decorrelated_dir, 'log-differenced'),
            os.path.join(features_dir, 'log-differenced'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'log-normalized'),
            os.path.join(features_dir, 'log-normalized'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'log-standardized'),
            os.path.join(features_dir, 'log-standardized'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'diff-normalized'),
            os.path.join(features_dir, 'diff-normalized'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'diff-standardized'),
            os.path.join(features_dir, 'diff-standardized'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'log-diff-normalized'),
            os.path.join(features_dir, 'log-diff-normalized'),
            nr_lags
            )

create_features_alt(
            os.path.join(decorrelated_dir, 'log-diff-standardized'),
            os.path.join(features_dir, 'log-diff-standardized'),
            nr_lags
            )

############################################

splits_dir = os.path.join(output_dir, '6-split')

split_data_alt(features_dir, splits_dir)

