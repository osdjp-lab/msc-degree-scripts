#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Set the directory containing the CSV files
input_dir = '../../1-preprocessing/data/rf/8-split/USD'

common_name = 'nr_trees'

# Define the hyperparameter grid
param_grid = {
    'n_estimators': np.arange(1, 15, 1)
}

train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
test_data = pd.read_csv(os.path.join(input_dir, "test_data.csv"))

# Split the data into features and target
X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

output_dir = "data"

os.makedirs(output_dir, exist_ok=True)

# Initialize the RandomForestRegressor model
model = RandomForestRegressor(
    # oob_score=True,
    random_state=0,
    n_jobs=-1
)

# Perform grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate R2
r2 = r2_score(y_test, y_pred)

# Calculate the Directional Symmetry (hit rate)
hit_rate = (np.sign(y_pred) == np.sign(y_test)).mean()

# Print the results
print(f"Best parameters: {grid_search.best_params_}")
print("---")
print(f"Best NMSE: {grid_search.best_score_:.3f}")
print("---")
print("Test set evaluation:")
print(f"MSE: {mse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R2: {r2:.3f}")
print(f"Directional Symmetry (hit rate): {hit_rate:.2f}")

# Save single parameter fit results

cv_results = grid_search.cv_results_

nr_trees = [d.get('n_estimators') for d in cv_results['params']]
score = cv_results['mean_test_score']

output_df = pd.concat([pd.Series(nr_trees, name='nr_trees'),
                       pd.Series(score, name="negmse")], axis=1)

result = os.path.join(output_dir, f"{common_name}_fit.csv")

output_df.to_csv(result, index=False)

# Save forecast

date = test_data['Date']
y_test.name = "y_test"

result = os.path.join(output_dir, f"{common_name}_forecast.csv")

output_df = pd.concat([date, y_test, pd.Series(y_pred, name="y_pred")], axis=1)
output_df.to_csv(result, index=False)

