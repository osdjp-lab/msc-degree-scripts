#!/usr/bin/env python3

# Linear decorrelation vs different preprocessing steps

import os
from preprocessing import *
from sklearn.svm import SVR

correlated_dir = 'data/decorrelation-tests/correlated'
os.makedirs(correlated_dir, exist_ok=True)

log_transform('data/2-forward-filled.csv',
              os.path.join(correlated_dir, '3-log-transformed.csv'))

difference('data/2-forward-filled.csv',
           os.path.join(correlated_dir, '3-differenced.csv'))

difference(os.path.join(correlated_dir, '3-log-transformed.csv'),
           os.path.join(correlated_dir, '3-log-differenced.csv'))

normalize("data/2-forward-filled.csv",
          os.path.join(correlated_dir, "3-normalized.csv"),
          (-1,1))

standardize("data/2-forward-filled.csv",
            os.path.join(correlated_dir, "3-standardized.csv"))

############################################

print("Creating groupings")

groupings = {"USD": ["JPY","CZK","DKK","GBP","HUF","PLN","SEK","CHF","NOK","AUD","CAD","HKD","KRW","NZD","SGD","ZAR"]}

groupings_dir = 'data/decorrelation-tests/groupings'
os.makedirs(groupings_dir, exist_ok=True)

create_groupings(os.path.join(correlated_dir, '3-log-transformed.csv'),
                 'data/decorrelation-tests/groupings/log-transformed',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-differenced.csv'),
                 'data/decorrelation-tests/groupings/differenced',
                 groupings)

create_groupings(os.path.join(correlated_dir, '3-log-differenced.csv'),
                 'data/decorrelation-tests/groupings/log-differenced',
                 groupings)

create_groupings('data/2-forward-filled.csv',
                 'data/decorrelation-tests/groupings/raw', groupings)

create_groupings(os.path.join(correlated_dir, "3-normalized.csv"),
                 'data/decorrelation-tests/groupings/normalized', groupings)

create_groupings(os.path.join(correlated_dir, "3-standardized.csv"),
                 'data/decorrelation-tests/groupings/standardized', groupings)

############################################

print("Decorrelating variables")

decorrelated_dir = 'data/decorrelation-tests/decorrelated'
os.makedirs(decorrelated_dir, exist_ok=True)

decorrelate(
            'data/decorrelation-tests/groupings/log-transformed',
            'data/decorrelation-tests/decorrelated/log-transformed'
            )

decorrelate(
            'data/decorrelation-tests/groupings/differenced',
            'data/decorrelation-tests/decorrelated/differenced'
            )

decorrelate(
            'data/decorrelation-tests/groupings/log-differenced',
            'data/decorrelation-tests/decorrelated/log-differenced'
            )

decorrelate(
            'data/decorrelation-tests/groupings/raw',
            'data/decorrelation-tests/decorrelated/raw'
            )

decorrelate(
            'data/decorrelation-tests/groupings/normalized',
            'data/decorrelation-tests/decorrelated/normalized'
            )

decorrelate(
            'data/decorrelation-tests/groupings/standardized',
            'data/decorrelation-tests/decorrelated/standardized'
            )
