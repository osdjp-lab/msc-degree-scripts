# Evaluate different time shifts

## Base goal

+ Save accuracy data for default model forecasts for training and test sets.
+ Plot forecasts on single graph for each dataset preprocessing methodology for each model.
+ For each model, for each dataset preprocessing methodology for the best forecasting shift save the best score and preprocessing methodology.
+ Pick the best set of data preprocessing steps and shifts for each model.

## Optimizations

+ Use only corresponding dataset:model sets or reduced mix
- Generate time lagged versions with different step for fewer models
- Export results for train and test sets into a single file

