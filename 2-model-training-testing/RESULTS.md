# Results

## MLP

Fitting 5 folds for each of 96 candidates, totalling 480 fits
Best parameters: {'activation': 'tanh', 'alpha': 0.1, 'hidden\_layer\_sizes': 50, 'learning\_rate\_init': 0.001, 'max\_iter': 1000, 'solver': 'adam'}
---
Best NMSE: -0.015
---
Test set evaluation:
MSE: 0.009
MAE: 0.080
MAPE: 150935472866.366
R2: 0.589
Directional Symmetry (hit rate): 0.94

Fitting 5 folds for each of 96 candidates, totalling 480 fits
Best parameters: {'activation': 'tanh', 'alpha': 0.1, 'hidden\_layer\_sizes': 50, 'learning\_rate\_init': 0.001, 'max\_iter': 1000, 'solver': 'adam'}
---
Best NMAE: -0.103
---
Test set evaluation:
MSE: 0.009
MAE: 0.080
MAPE: 150935472866.366
R2: 0.589
Directional Symmetry (hit rate): 0.94

Fitting 5 folds for each of 96 candidates, totalling 480 fits
Best parameters: {'activation': 'identity', 'alpha': 0.1, 'hidden\_layer\_sizes': 100, 'learning\_rate\_init': 0.01, 'max\_iter': 1000, 'solver': 'sgd'}
---
Best NMAPE: -1.017
---
Test set evaluation:
MSE: 0.034
MAE: 0.137
MAPE: 33023832330.081
R2: -0.567
Directional Symmetry (hit rate): 0.86

Fitting 5 folds for each of 96 candidates, totalling 480 fits
Best parameters: {'activation': 'tanh', 'alpha': 0.1, 'hidden\_layer\_sizes': 50, 'learning\_rate\_init': 0.001, 'max\_iter': 1000, 'solver': 'adam'}
---
Best R2: 0.626
---
Test set evaluation:
MSE: 0.009
MAE: 0.080
MAPE: 150935472866.366
R2: 0.589
Directional Symmetry (hit rate): 0.94

## SVR

Fitting 5 folds for each of 270 candidates, totalling 1350 fits
Best parameters: {'C': 1, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'tol': 0.01}
---
Best NMSE: -0.312
---
Test set evaluation:
MSE: 0.120
MAE: 0.260
MAPE: 1.905
R2: 0.109
Directional Symmetry (hit rate): 0.90

Fitting 5 folds for each of 270 candidates, totalling 1350 fits
Best parameters: {'C': 1, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'tol': 0.1}
---
Best NMAE: -0.414
---
Test set evaluation:
MSE: 0.141
MAE: 0.282
MAPE: 1.969
R2: -0.045
Directional Symmetry (hit rate): 0.88

Fitting 5 folds for each of 270 candidates, totalling 1350 fits
Best parameters: {'C': 1, 'epsilon': 0.1, 'gamma': 'auto', 'kernel': 'rbf', 'tol': 1e-05}
---
Best NMAPE: -1.830
---
Test set evaluation:
MSE: 0.165
MAE: 0.309
MAPE: 2.097
R2: -0.224
Directional Symmetry (hit rate): 0.84

Fitting 5 folds for each of 270 candidates, totalling 1350 fits
Best parameters: {'C': 1, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf', 'tol': 0.1}
---
Best R2: 0.313
---
Test set evaluation:
MSE: 0.141
MAE: 0.282
MAPE: 1.969
R2: -0.045
Directional Symmetry (hit rate): 0.88

## RF

Fitting 5 folds for each of 288 candidates, totalling 1440 fits
Best parameters: {'max\_depth': 10, 'max\_features': 0.3, 'min\_samples\_leaf': 2, 'min\_samples\_split': 5, 'n\_estimators': 100}
---
Best NMSE: -0.020
---
Test set evaluation:
MSE: 0.012
MAE: 0.098
MAPE: 0.090
R2: -2.805
Directional Symmetry (hit rate): 1.00

Fitting 5 folds for each of 288 candidates, totalling 1440 fits
Best parameters: {'max\_depth': 10, 'max\_features': 0.3, 'min\_samples\_leaf': 2, 'min\_samples\_split': 5, 'n\_estimators': 100}
---
Best NMAE: -0.106
---
Test set evaluation:
MSE: 0.012
MAE: 0.098
MAPE: 0.090
R2: -2.805
Directional Symmetry (hit rate): 1.00

Fitting 5 folds for each of 288 candidates, totalling 1440 fits
Best parameters: {'max\_depth': 10, 'max\_features': 0.3, 'min\_samples\_leaf': 2, 'min\_samples\_split': 5, 'n\_estimators': 100}
---
Best NMAPE: -0.092
---
Test set evaluation:
MSE: 0.012
MAE: 0.098
MAPE: 0.090
R2: -2.805
Directional Symmetry (hit rate): 1.00

Fitting 5 folds for each of 288 candidates, totalling 1440 fits
Best parameters: {'max\_depth': 10, 'max\_features': 0.3, 'min\_samples\_leaf': 2, 'min\_samples\_split': 5, 'n\_estimators': 100}
---
Best R2: -0.800
---
Test set evaluation:
MSE: 0.012
MAE: 0.098
MAPE: 0.090
R2: -2.805
Directional Symmetry (hit rate): 1.00

