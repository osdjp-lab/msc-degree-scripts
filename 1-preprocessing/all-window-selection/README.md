# Evaluate different time shifts

## Base goal

+ Save accuracy data for default model forecasts for training and test sets.
- Plot forecasts on single graph for each dataset preprocessing methodology.
- For each model, for each dataset preprocessing methodology for the best forecasting shift export a single file containing the best score and preprocessing path.
- Pick the best set of data preprocessing steps and shifts for each model.

## Optimizations

- Use only corresponding dataset:model sets
- Generate time lagged versions with different step for fewer models

