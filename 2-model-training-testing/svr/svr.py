#!/usr/bin/env python3

# Results:
# Fitting 5 folds for each of 270 candidates, totalling 1350 fits
# Best parameters: {'C': 1, 'epsilon': 0.1, 'gamma': 'auto', 'kernel': 'rbf', 'tol': 1e-05}
# ---
# Best  NMSE: -1.830
# ---
# Test set evaluation:
# MAPE: 2.097
# Directional Symmetry (hit rate): 0.84

import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Set the directory containing the CSV files
input_dir = '../../1-preprocessing/data/svr/8-split/USD'

# Define the hyperparameter grid
param_grid = {
    'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    'C': [1, 10, 100],
    'epsilon': [0.1, 0.01, 0.001],
    'gamma': ['scale','auto'],
    'kernel': ['rbf','poly','sigmoid']
}

# Optimized
# param_grid = {
#     'tol': [1e-5],
#     'C': [1],
#     'epsilon': [0.1],
#     'gamma': ['auto'],
#     'kernel': ['rbf']
# }

train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
test_data = pd.read_csv(os.path.join(input_dir, "test_data.csv"))

# Split the data into features and target
X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

scoring_map = {'neg_mean_squared_error': 'NMSE',
               'neg_mean_absolute_error': 'NMAE',
               'neg_mean_absolute_percentage_error': 'NMAPE',
               'r2': 'R2'}

output_dir = "svr-forecasts"

os.makedirs(output_dir, exist_ok=True)

for scoring in ['neg_mean_squared_error',
                'neg_mean_absolute_error',
                'neg_mean_absolute_percentage_error',
                'r2']:
           
    # Initialize the SVR model
    model = SVR()
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring=scoring,
        n_jobs=-1,  # use all available CPU cores
        verbose=3
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Calculate R2
    r2 = r2_score(y_test, y_pred)
    
    # Calculate the Directional Symmetry (hit rate)
    hit_rate = (np.sign(y_pred) == np.sign(y_test)).mean()
    
    # Print the results
    print(f"Best parameters: {grid_search.best_params_}")
    print("---")
    print(f"Best {scoring_map[scoring]}: {grid_search.best_score_:.3f}")
    print("---")
    print("Test set evaluation:")
    print(f"MSE: {mse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"MAPE: {mape:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Directional Symmetry (hit rate): {hit_rate:.2f}")
    print()
    
    date = test_data['Date']
    y_test.name = "y_test"

    result = os.path.join(output_dir, f"{scoring_map[scoring]}.csv")
    
    output_df = pd.concat([date, y_test, pd.Series(y_pred, name="y_pred")], axis=1)
    output_df.to_csv(result, index=False)

