#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

# Set the directory containing the CSV files
input_dir = '../../1-preprocessing/data/rf/8-split/USD'

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_features': [1.0, 'sqrt', 'log2', 0.3],
    'max_depth': [1, 3, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Optimized
# param_grid = {
#     'n_estimators': [100],
#     'max_features': ['sqrt'],
#     'max_depth': [1],
#     'min_samples_split': [2],
#     'min_samples_leaf': [2]
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

output_dir = "rf-forecasts"

os.makedirs(output_dir, exist_ok=True)

for scoring in ['neg_mean_squared_error',
                'neg_mean_absolute_error',
                'neg_mean_absolute_percentage_error',
                'r2']:

    # Initialize model
    model = RandomForestRegressor(
        # oob_score=True,
        random_state=0
    )
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring=scoring,
        n_jobs=-1,  # use all available CPU cores
        verbose=3
    )
    
    # Fit the RandomForestRegressor model
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
    # oob_error = 1 - grid_search.best_estimator_.oob_score_
    # print(f"Best  OOB Error: {oob_error}")
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

