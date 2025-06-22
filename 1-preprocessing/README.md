# Preprocessing scripts

## Requirements

- python 3
- numpy
- pandas
- scikit-learn
- csvkit
- statsmodels

## TODO

+ Pearson Correlation matrix (What should be kept? What should be rejected? How to interpret the results?)
+ Selection of currency sets (n -> 1 forecasting)
+ Selection of exchange rate predictors and targets (Which with which? -> Those currencies with the lowest linear correlation with the target variable)
+ How many predictors should be selected? (5)
+ What offsets (time lags) should be applied to the predictors (All the same? Different for each?)? (a single time lag for all)
+ Single split: only training and test set (split of training set for cross-validation done via scikit-learn models)

## Other

- Reasoning order for failure of model: Data (lack of predictive potential of data) -> Model training -> Parameter selection
- If prediction model still has terrible accuracy (add other data: indexes, stocks - only upto 5)
- Statistical analysis of the dataset
- Pearson Correlation matrixes for different offsets (time lags) for different currencies
- Create a Pearson correlation coefficient percentage of values above graph for easier choice of currency rejection value

