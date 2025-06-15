#!/usr/bin.env sh

# Select only columns with no missing data

csvcut -c 1,2,3,6,7,9,10,14,17,20,22,27,29,31,35,38,40,42 data/0-raw.csv > data/1-selected.csv

