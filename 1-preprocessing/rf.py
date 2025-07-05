#!/usr/bin/env python3

# Prepare datasets for Random Forest

import os
import shutil
from preprocessing import *
from sklearn.svm import SVR

output_dir = "data/rf/"

os.makedirs(output_dir, exist_ok=True)

src = "data/2-forward-filled.csv"

dst = os.path.join(output_dir, "3-none.csv")

shutil.copy(src, dst)

groupings = {"USD": ["JPY","CZK","DKK","GBP","HUF","PLN","SEK","CHF","NOK","AUD","CAD","HKD","KRW","NZD","SGD","ZAR"]}

print("Creating groupings")

create_groupings(os.path.join(output_dir, "3-none.csv"),
                 os.path.join(output_dir, "4-groupings/"),
                 groupings)

time_lags = 50

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

