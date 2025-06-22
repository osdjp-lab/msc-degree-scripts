# Issues

## Preprocessing performance issues

- Static feature-target dataset generation (time lags)
- Static train-test split generation
- Normalization range (-1,1) [tanh activation function] - possibility of better results for reduced range (-0.8,0.8)

## Forecasting methodology

- Size of sample window 'n' for 1 to 'n' time lags of 'm' exchange rates
- Offset of sample window in regards to target variable values (default: -1)
- Sample window set to 1 for 'm' variables and variable offset

## Prediction methodology

- Based on test set predictor values (iterative)
- Based on initial test set predictor set of values followed by the models generated result for consecutive values
- Model performance better on test set than training set

## Potential issues

- Too large dataset (use subset) [no effect]
- Unsuitable exchange rate predictors

