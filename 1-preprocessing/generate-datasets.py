#!/usr/bin/env python3

import os
from preprocessing import *

input_dir = '../data/0-raw'
output_dir = '../data/1-preprocessing'
os.makedirs(output_dir, exist_ok=True)

# Common handling of missing values

remove_variables_with_missing_values(
        os.path.join(input_dir, 'eurofxref.csv'),
        os.path.join(output_dir, '1-removed-nan-variables.csv'))

interpolate(
        os.path.join(output_dir, '1-removed-nan-variables.csv'),
        os.path.join(output_dir, '2-interpolated.csv'))

# All combinations of preprocessing steps

correlated_dir = os.path.join(output_dir, '3-correlated')
os.makedirs(correlated_dir, exist_ok=True)

# Raw

log_transform(os.path.join(output_dir, '2-interpolated.csv'),
              os.path.join(correlated_dir, 'log-transformed.csv'))

difference(os.path.join(output_dir, '2-interpolated.csv'),
           os.path.join(correlated_dir, 'differenced.csv'))

normalize(os.path.join(output_dir, '2-interpolated.csv'),
          os.path.join(correlated_dir, 'normalized.csv'),
          (-1,1))

standardize(os.path.join(output_dir, '2-interpolated.csv'),
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

groupings_dir = os.path.join(output_dir, '4-groupings')
os.makedirs(groupings_dir, exist_ok=True)

create_groupings(os.path.join(output_dir, '2-interpolated.csv'),
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

decorrelated_dir = os.path.join(output_dir, '5-decorrelated')
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

features_dir = os.path.join(output_dir, '6-features')
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

splits_dir = os.path.join(output_dir, '7-split')

split_data_alt(features_dir, splits_dir)

