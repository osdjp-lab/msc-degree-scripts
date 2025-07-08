# Notes

Feature scaling (normalization/standardization)

For each currency (time series) the time lagged versions of the currency have a high correlation with each other over 0.99. As such there are only two solutions:
1. Take only one time lagged variable for each time series (preposition: since each time lagged version is nearly identical to the non time lagged version) picking only the least correlated time lag for each exchange rate.
2. See how far back the auto correlated behaviour of the time series reaches 200 days, 2000 days, 20000 days to see if there is a time lag where the currency no longer is highly correlated with itself.

Decorrelation vs preprocessing (differencing, log transformation)

