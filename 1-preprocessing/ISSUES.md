# Issues

## Preprocessing step selection

1. Minimal required preprocessing steps for each model.
2. Evaluate initial datasets using default model parameters.
3. Apply and evaluaute different combinations of optional preprocessing steps for each model.

## Necessary vs required preprocessing steps for each model

### Note

Feature scaling (normalization/standardization)

For each currency (time series) the time lagged versions of the currency have a high correlation with each other over 0.99. As such there are only two solutions:
1. Take only one time lagged variable for each time series (preposition: since each time lagged version is nearly identical to the non time lagged version) picking only the least correlated time lag for each exchange rate.
2. See how far back the auto correlated behaviour of the time series reaches 200 days, 2000 days, 20000 days to see if there is a time lag where the currency no longer is highly correlated with itself.

### Required

Common prior:
- Handling of missing values

Specific:
- Neural network: Normalization
- SVR: Standardization (mean = 0; variance = 1); Handling of outliers (removal of outliers or windsoring) 
- Random forest: None

Post:
- Feature creation (time lags for all variables)
- Correlation analysis (covariance, cointegration, chi-square, linear correlation)
- Feature selection (irrelevant or redundant features should be removed) - possibly via RFE, RFECV or Random Forest
- Train/test split

### Optional

- Log transformation
- Differencing

## Preprocessing performance issues

+ Static feature-target dataset generation (time lags) - only generate largest time lag (and change loaded columns during training)
+ Static train-test split generation (move to training)

## Preprocessing considerations

+ Log transformation
+ Information leakage between train and test sets (add separation)
- Statistical analysis of the dataset
- Check performance when omitting certain steps
- Winsoring or Box-Cox transformation of the data
- Cointegration
- Regularization (L1, L2, dropout, early stopping)
- Choice of better predictors: indexes, stocks, etc.
- Pearson Correlation matrixes for different offsets (time lags) for different currencies
- Create a Pearson correlation coefficient percentage of values above, graph for easier choice of currency rejection value
- Exponential smoothing and technical indicators [prediction]
- Normalization range (-1,1) [tanh activation function] - possibility of better results for reduced range (-0.8,0.8) [nn-vs-chaotic]
- Descriptive statistics, Bera-Jarque test, Autocorrelation coefficients, Bartlett standard errors, Ljung-Box Q statistic [robust-regression]

