#!/usr/bin/env python3

# Handle missing values

from preprocessing import remove_variables_with_missing_values
from preprocessing import forward_fill

remove_variables_with_missing_values(
        "data/0-raw.csv",
        "data/1-removed-nan-variables.csv")

forward_fill(
        "data/1-removed-nan-variables.csv",
        "data/2-forward-filled.csv")

