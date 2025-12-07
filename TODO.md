# TODO

## Main

+ Add README.md for raw data sources
+ Add suitable datasets
- Add dataset merge step
- Add preprocessing of added datasets
- Review necessity of decorrelation preprocessing step
- Cherry pick or copy paste and re-add switch from forward fill to interpolate
- Review different data frequencies (daily, monthly, quarterly)
- Add frequency based preprocessing split - multiple paths: daily (monthly, quarterly -> daily) and monthly, quarterly (daily -> monthly, quarterly)

## Potential

- Make scripts location independant (no hardcoded paths) - maybe split preprocessing library into individual scripts
- Split each dataset into virtual interpolated subdatasets based on different sampling frequency subdatasets (Taylor series expansion)

## Dropped

- Add Gold, Oil, GDP, Inflation etc. datasets (how are they represented - used currency - and how to integrate them)

