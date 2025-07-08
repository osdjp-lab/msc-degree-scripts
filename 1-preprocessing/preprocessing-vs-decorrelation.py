#!/usr/bin/env python3

# Linear decorrelation vs different preprocessing steps

import os
from preprocessing import *
from sklearn.svm import SVR

correlated_dir = 'data/decorrelation-tests/1-correlated'
os.makedirs(correlated_dir, exist_ok=True)

# Raw

log_transform('data/2-forward-filled.csv',
              os.path.join(correlated_dir, '3-log-transformed.csv'))

difference('data/2-forward-filled.csv',
           os.path.join(correlated_dir, '3-differenced.csv'))

normalize("data/2-forward-filled.csv",
          os.path.join(correlated_dir, "3-normalized.csv"),
          (-1,1))

standardize("data/2-forward-filled.csv",
            os.path.join(correlated_dir, "3-standardized.csv"))

# 1st degree combinations

# Logs

difference(os.path.join(correlated_dir, '3-log-transformed.csv'),
           os.path.join(correlated_dir, '3-log-differenced.csv'))

normalize(os.path.join(correlated_dir, '3-log-transformed.csv'),
          os.path.join(correlated_dir, '3-log-normalized.csv'))

standardize(os.path.join(correlated_dir, '3-log-transformed.csv'),
            os.path.join(correlated_dir, '3-log-standardized.csv'))

# Diffs

normalize(os.path.join(correlated_dir, '3-differenced.csv'),
          os.path.join(correlated_dir, '3-diff-normalized.csv'))

standardize(os.path.join(correlated_dir, '3-differenced.csv'),
            os.path.join(correlated_dir, '3-diff-standardized.csv'))

# 2nd degree combinations

normalize(os.path.join(correlated_dir, '3-log-differenced.csv'),
          os.path.join(correlated_dir, '3-log-diff-normalized.csv'))

standardize(os.path.join(correlated_dir, '3-log-differenced.csv'),
            os.path.join(correlated_dir, '3-log-diff-standardized.csv'))

############################################

print("Creating groupings")

groupings = {"USD": ["JPY","CZK","DKK","GBP","HUF","PLN","SEK","CHF","NOK","AUD","CAD","HKD","KRW","NZD","SGD","ZAR"]}

groupings_dir = 'data/decorrelation-tests/2-groupings'
os.makedirs(groupings_dir, exist_ok=True)

create_groupings('data/2-forward-filled.csv',
                 'data/decorrelation-tests/groupings/raw', groupings)

create_groupings(os.path.join(correlated_dir, '3-log-transformed.csv'),
                 'data/decorrelation-tests/groupings/log-transformed',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-differenced.csv'),
                 'data/decorrelation-tests/groupings/differenced',
                 groupings)

create_groupings(os.path.join(correlated_dir, "3-normalized.csv"),
                 'data/decorrelation-tests/groupings/normalized', groupings)

create_groupings(os.path.join(correlated_dir, "3-standardized.csv"),
                 'data/decorrelation-tests/groupings/standardized', groupings)

# Combinations

create_groupings(os.path.join(correlated_dir, '3-log-differenced.csv'),
                 'data/decorrelation-tests/groupings/log-differenced',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-log-normalized.csv'),
                 'data/decorrelation-tests/groupings/log-normalized',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-log-standardized.csv'),
                 'data/decorrelation-tests/groupings/log-standardized',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-diff-normalized.csv'),
                 'data/decorrelation-tests/groupings/diff-normalized',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-diff-standardized.csv'),
                 'data/decorrelation-tests/groupings/diff-standardized',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-log-diff-normalized.csv'),
                 'data/decorrelation-tests/groupings/log-diff-normalized',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-log-diff-standardized.csv'),
                 'data/decorrelation-tests/groupings/log-diff-standardized',
                 groupings)

############################################

print("Decorrelating variables")

decorrelated_dir = 'data/decorrelation-tests/3-decorrelated'
os.makedirs(decorrelated_dir, exist_ok=True)

decorrelate(
            'data/decorrelation-tests/groupings/raw',
            'data/decorrelation-tests/decorrelated/raw'
            )

decorrelate(
            'data/decorrelation-tests/groupings/log-transformed',
            'data/decorrelation-tests/decorrelated/log-transformed'
            )

decorrelate(
            'data/decorrelation-tests/groupings/differenced',
            'data/decorrelation-tests/decorrelated/differenced'
            )

decorrelate(
            'data/decorrelation-tests/groupings/normalized',
            'data/decorrelation-tests/decorrelated/normalized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/standardized',
            'data/decorrelation-tests/decorrelated/standardized'
            )

# Combinations

decorrelate(
            'data/decorrelation-tests/groupings/log-differenced',
            'data/decorrelation-tests/decorrelated/log-differenced'
            )

decorrelate(
            'data/decorrelation-tests/groupings/log-normalized',
            'data/decorrelation-tests/decorrelated/log-normalized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/log-standardized',
            'data/decorrelation-tests/decorrelated/log-standardized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/diff-normalized',
            'data/decorrelation-tests/decorrelated/diff-normalized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/diff-standardized',
            'data/decorrelation-tests/decorrelated/diff-standardized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/log-diff-normalized',
            'data/decorrelation-tests/decorrelated/log-diff-normalized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/log-diff-standardized',
            'data/decorrelation-tests/decorrelated/log-diff-standardized'
            )

############################################

# Get time lags

features_dir = 'data/decorrelation-tests/4-features'
os.makedirs(features_dir, exist_ok=True)

nr_lags = 50

create_features_alt(
            'data/decorrelation-tests/decorrelated/raw',
            'data/decorrelation-tests/features/raw',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/log-transformed',
            'data/decorrelation-tests/features/log-transformed',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/differenced',
            'data/decorrelation-tests/features/differenced',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/normalized',
            'data/decorrelation-tests/features/normalized',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/standardized',
            'data/decorrelation-tests/features/standardized',
            nr_lags
            )

# Combinations

create_features_alt(
            'data/decorrelation-tests/decorrelated/log-differenced',
            'data/decorrelation-tests/features/log-differenced',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/log-normalized',
            'data/decorrelation-tests/features/log-normalized',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/log-standardized',
            'data/decorrelation-tests/features/log-standardized',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/diff-normalized',
            'data/decorrelation-tests/features/diff-normalized',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/diff-standardized',
            'data/decorrelation-tests/features/diff-standardized',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/log-diff-normalized',
            'data/decorrelation-tests/features/log-diff-normalized',
            nr_lags
            )

create_features_alt(
            'data/decorrelation-tests/decorrelated/log-diff-standardized',
            'data/decorrelation-tests/features/log-diff-standardized',
            nr_lags
            )

############################################

# Select window

window_dir = 'data/decorrelation-tests/features'
os.makedirs(window_dir, exist_ok=True)



