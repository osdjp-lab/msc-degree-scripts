#!/usr/bin/env python3

# Prepare datasets for Multi Layer Perceptron

# Note
# 50 - 1060 s = 17 min 40 s

import os
from preprocessing import normalize, create_groupings, create_features, select_features
from sklearn.svm import SVR

output_dir = "data/nn/"

# Create the output directory if it doesn"t exist
os.makedirs(output_dir, exist_ok=True)

print("Normalizing")

normalize("data/2-forward-filled.csv",
          os.path.join(output_dir, "3-normalized.csv"),
          (-1,1))

groupings = {"USD": ["JPY","CZK","DKK","GBP","HUF","PLN","SEK","CHF","NOK","AUD","CAD","HKD","KRW","NZD","SGD","ZAR"]}

print("Creating groupings")

create_groupings(os.path.join(output_dir, "3-normalized.csv"),
                 os.path.join(output_dir, "4-groupings/"),
                 groupings)

time_lags = 50

print("Creating features")

create_features(os.path.join(output_dir, "4-groupings/"),
                os.path.join(output_dir, "5-features/"),
                time_lags)

print("Running feature selection")

print("SVR")

estimator = SVR(kernel="linear")

select_features(
       estimator,
       os.path.join(output_dir, "5-features/"),
       os.path.join(output_dir, "6-svr-selection"))

