# Issues

## General

+ Use other forecast evaluation metrics (R2 sensitive to outliers) - MSE, RMSE, MAE
+ Evaluate models forecasting performance on the training dataset (bias-variance dilemma)
- Use baseline models for reference: ARIMA, VAR, random walk 

## Prediction methodology

- Based on test set predictor values (iterative)
- Based on initial test set predictor set of values followed by the models generated result for consecutive values - only for univariate forecasting.

## Potential issues

- Model performance better on test set than training set
- Too large dataset (use subset) [no effect]
- Unsuitable exchange rate predictors

## Optimization strategies

+ Use small subset of data for faster initial parameter selection (unapplicable - significant variation of results)

