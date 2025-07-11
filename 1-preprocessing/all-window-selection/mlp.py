#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

input_dir = '../data/decorrelation-tests/5-split'
output_dir = 'data/mlp'
os.makedirs(output_dir, exist_ok=True)

for subdir in os.listdir(input_dir):
    if 'normalized' in subdir:
        for target in os.listdir(os.path.join(input_dir, subdir)):
            rel_path = os.path.join(subdir, target)
            
            train_mse_results = pd.DataFrame(columns=['offset', 'mse'])
            train_mae_results = pd.DataFrame(columns=['offset', 'mae'])
            train_r2_results = pd.DataFrame(columns=['offset', 'r2'])
            train_hitrate_results = pd.DataFrame(columns=['offset', 'hitrate'])
            
            test_mse_results = pd.DataFrame(columns=['offset', 'mse'])
            test_mae_results = pd.DataFrame(columns=['offset', 'mae'])
            test_r2_results = pd.DataFrame(columns=['offset', 'r2'])
            test_hitrate_results = pd.DataFrame(columns=['offset', 'hitrate'])

           
            result_dir = os.path.join(output_dir, rel_path)
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(os.path.join(result_dir, 'forecasts'), exist_ok=True)
            
            print(result_dir)

            for offset in os.listdir(os.path.join(input_dir, rel_path)):
                split_input_dir = os.path.join(input_dir, rel_path, offset)

                print(split_input_dir)
                
                train_data = pd.read_csv(os.path.join(split_input_dir, "train_data.csv"))
                test_data = pd.read_csv(os.path.join(split_input_dir, "test_data.csv"))

                # Split the data into features and target
                train_date = train_data.iloc[:,0]
                X_train = train_data.iloc[:, 1:-1]
                y_train = train_data.iloc[:, -1]
                
                test_date = test_data.iloc[:,0]
                X_test = test_data.iloc[:, 1:-1]
                y_test = test_data.iloc[:, -1]
                
                # Fit the MLPRegressor model
                model = MLPRegressor(activation="tanh",
                                     shuffle=False,
                                     random_state=0,
                                     max_iter=1000)
                
                model.fit(X_train, y_train)
                
                # Training set prediction evaluation
            
                # Predict the training set
                y_train_pred = model.predict(X_train)
           
                # Calculate scoring metrics
                train_mse = mean_squared_error(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                if 'diff' in subdir:
                    train_hitrate = (np.sign(y_train_pred) == np.sign(y_train)).mean()
                else:
                    train_hitrate = (np.sign(pd.Series(y_train_pred, name='y_train_pred').diff()) == np.sign(y_train.diff())).mean()
            
                # Add the results to the DataFrames
                train_mse_results = pd.concat([train_mse_results, pd.DataFrame({'offset': [offset], 'mse': [train_mse]})])
                train_mae_results = pd.concat([train_mae_results, pd.DataFrame({'offset': [offset], 'mae': [train_mae]})])
                train_r2_results = pd.concat([train_r2_results, pd.DataFrame({'offset': [offset], 'r2': [train_r2]})])
                train_hitrate_results = pd.concat([train_hitrate_results, pd.DataFrame({'offset': [offset], 'hitrate': [train_hitrate]})])
               
                # Test set prediction evaluation
            
                # Predict the test set
                y_test_pred = model.predict(X_test)
            
                # Calculate scoring metrics
                test_mse = mean_squared_error(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                if 'diff' in subdir:
                    test_hitrate = (np.sign(y_test_pred) == np.sign(y_test)).mean()
                else:
                    test_hitrate = (np.sign(pd.Series(y_test_pred, name='y_test_pred').diff()) == np.sign(y_test.diff())).mean()
            
                # Add the results to the DataFrames
                test_mse_results = pd.concat([test_mse_results, pd.DataFrame({'offset': [offset], 'mse': [test_mse]})])
                test_mae_results = pd.concat([test_mae_results, pd.DataFrame({'offset': [offset], 'mae': [test_mae]})])
                test_r2_results = pd.concat([test_r2_results, pd.DataFrame({'offset': [offset], 'r2': [test_r2]})])
                test_hitrate_results = pd.concat([test_hitrate_results, pd.DataFrame({'offset': [offset], 'hitrate': [test_hitrate]})])

                # Save forecasts
                output_df = pd.concat([train_date, pd.Series(y_train, name='y_train'), pd.Series(y_train_pred, name="y_train_pred")], axis=1)
                output_df.to_csv(os.path.join(result_dir, 'forecasts', f'{offset}_train_pred.csv'), index=False)
                output_df = pd.concat([test_date, pd.Series(y_test, name='y_test'), pd.Series(y_test_pred, name="y_test_pred")], axis=1)
                output_df.to_csv(os.path.join(result_dir, 'forecasts', f'{offset}_test_pred.csv'), index=False)

            train_mse_results.to_csv(os.path.join(result_dir, 'train_mse_results.csv'), index=False)
            train_mae_results.to_csv(os.path.join(result_dir, 'train_mae_results.csv'), index=False)
            train_r2_results.to_csv(os.path.join(result_dir, 'train_r2_results.csv'), index=False)
            train_hitrate_results.to_csv(os.path.join(result_dir, 'train_hitrate_results.csv'), index=False)
            
            test_mse_results.to_csv(os.path.join(result_dir, 'test_mse_results.csv'), index=False)
            test_mae_results.to_csv(os.path.join(result_dir, 'test_mae_results.csv'), index=False)
            test_r2_results.to_csv(os.path.join(result_dir, 'test_r2_results.csv'), index=False)
            test_hitrate_results.to_csv(os.path.join(result_dir, 'test_hitrate_results.csv'), index=False)
                
            print('Set complete')

print('Calculations complete')
        
