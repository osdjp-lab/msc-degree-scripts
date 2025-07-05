#!/usr/bin/env python3

# Prepare datasets for Support Vector Regression

# Notes
# 5  - 10^27 - 225 s = 3 min 45 s
# 6  - 10^32 - 322 s = 5 min 22 s
# 10 - 10^54 - 930 s = 15 min 30 s

import os
from preprocessing import *
from sklearn.svm import SVR

output_dir = "data/svr"

# Create the output directory if it doesn"t exist
os.makedirs(output_dir, exist_ok=True)

standardize("data/2-forward-filled.csv",
            os.path.join(output_dir, "3-standardized.csv"))

groupings = {"USD": ["JPY","CZK","DKK","GBP","HUF","PLN","SEK","CHF","NOK","AUD","CAD","HKD","KRW","NZD","SGD","ZAR"]}

print("Creating groupings")

create_groupings(os.path.join(output_dir, "3-standardized.csv"),
                 os.path.join(output_dir, "4-groupings/"),
                 groupings)

time_lags = 500

print("Creating features")

create_features(os.path.join(output_dir, "4-groupings/"),
                os.path.join(output_dir, "5-features/"),
                time_lags)

print("Decorrelating variables")

decorrelate(os.path.join(output_dir, "5-features/"),
            os.path.join(output_dir, "6-decorrelated/"))

print("Running feature selection")

print("SVR")

estimator = SVR(kernel="linear")

select_features(
       estimator,
       os.path.join(output_dir, "6-decorrelated/"),
       os.path.join(output_dir, "7-svr-selection"))

