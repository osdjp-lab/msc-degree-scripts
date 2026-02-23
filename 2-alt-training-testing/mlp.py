#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import optuna
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##################################################

model = MLPRegressor()

param_distributions = {
    'hidden_layer_sizes': optuna.distributions.IntDistribution(1, 30),
    'solver': optuna.distributions.CategoricalDistribution('adam','sgd','lbfgs'),
    'alpha': optuna.distributions.FloatDistribution(1e-10, 1e10, log=True),
    'tol': optuna.distributions.FloatDistribution(1e-10, 1e10, log=True)
}

optuna_search = optuna.integration.OptunaSearchCV(
    model, param_distributions, n_trials=1000, verbose=2
)

X, y = load_iris(return_X_y=True)
optuna_search.fit(X, y)

print("Best trial:")
trial = optuna_search.study_.best_trial

print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


##################################################

# Set the directory containing the CSV files
with open("../../data/3-training-testing/mlp/optimal_dataset_dir.txt", 'r') as file:
    input_dir = file.read().rstrip()

common_name = 'all'

output_dir = "../../data/3-training-testing/mlp"

os.makedirs(output_dir, exist_ok=True)

# Define the hyperparameter grid
param_grid = {
    # Initial search
    'hidden_layer_sizes': np.arange(1, 30, 1),
    'solver': ['adam','sgd','lbfgs'],
    'alpha': np.logspace(-10, -1, 10),
    'tol': np.logspace(-10, -1, 10),
}

train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
test_data = pd.read_csv(os.path.join(input_dir, "test_data.csv"))

# Split the data into features and target
X_train = train_data.iloc[:, 1:-1]
y_train = train_data.iloc[:, -1]
X_test = test_data.iloc[:, 1:-1]
y_test = test_data.iloc[:, -1]

# Initialize the MLPRegressor model
model = MLPRegressor(activation='tanh',
                     shuffle=False,
                     random_state=0,
                     max_iter=2000)

# Perform grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=-1,  # use all available CPU cores
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

# Save forecast

date = test_data['Date']
y_test.name = "y_test"

result = os.path.join(output_dir, f"{common_name}_forecast.csv")

output_df = pd.concat([date, y_test, pd.Series(y_pred, name="y_pred")], axis=1)
output_df.to_csv(result, index=False)

# Save all results

cv_results = grid_search.cv_results_

df = pd.DataFrame(cv_results)
df.columns = ['Param ' + col if col.startswith('param_') else col for col in df.columns]

df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)

