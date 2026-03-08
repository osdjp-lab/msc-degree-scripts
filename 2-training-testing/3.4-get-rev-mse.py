#!/usr/bin/env python3

# Calculate MSE, MAE, R2 and Hitrate for reverse transformed forecasts

import os
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

INPUT_DIR = Path("../data/2-training-testing")

for model in os.listdir(INPUT_DIR):

    model_path = INPUT_DIR / model
    
    for dataset_type in os.listdir(model_path):
        if not os.path.isdir(model_path / dataset_type): 
            continue
        
        dataset_path = model_path / dataset_type
            
        for target in os.listdir(dataset_path):
            if not os.path.isdir(dataset_path / target): 
                continue

            target_path = dataset_path / target

            # Containers for aggregated metrics (reset for each target)
            train_mse_results = pd.DataFrame(columns=["offset", "mse"])
            train_mae_results = pd.DataFrame(columns=["offset", "mae"])
            train_r2_results = pd.DataFrame(columns=["offset", "r2"])
            train_hitrate_results = pd.DataFrame(columns=["offset", "hitrate"])

            test_mse_results = pd.DataFrame(columns=["offset", "mse"])
            test_mae_results = pd.DataFrame(columns=["offset", "mae"])
            test_r2_results = pd.DataFrame(columns=["offset", "r2"])
            test_hitrate_results = pd.DataFrame(columns=["offset", "hitrate"])

            for offset in os.listdir(target_path):
                if not os.path.isdir(target_path / offset): 
                    continue

                offset_path = target_path / offset

                print(offset_path)

                # --------------------------------------------------------------
                # Load data
                # --------------------------------------------------------------
                test_set_rev_forecast = pd.read_csv(offset_path / "test_pred_rev.csv")
                train_set_rev_forecast = pd.read_csv(offset_path / "train_pred_rev.csv")

                y_test = test_set_rev_forecast.iloc[:, 1]
                y_test_pred = test_set_rev_forecast.iloc[:, 2]

                y_train = train_set_rev_forecast.iloc[:, 1]
                y_train_pred = train_set_rev_forecast.iloc[:, 2]

                # --------------------------------------------------------------
                # Evaluation on training set
                # --------------------------------------------------------------

                train_mse = mean_squared_error(y_train, y_train_pred)
                train_mae = mean_absolute_error(y_train, y_train_pred)
                train_r2 = r2_score(y_train, y_train_pred)

                train_hitrate = (
                    np.sign(pd.Series(y_train_pred, name="y_train_pred").diff())
                    == np.sign(y_train.diff())
                ).mean()

                # Append training metrics
                train_mse_results = pd.concat(
                    [train_mse_results, pd.DataFrame({"offset": [offset], "mse": [train_mse]})]
                )
                train_mae_results = pd.concat(
                    [train_mae_results, pd.DataFrame({"offset": [offset], "mae": [train_mae]})]
                )
                train_r2_results = pd.concat(
                    [train_r2_results, pd.DataFrame({"offset": [offset], "r2": [train_r2]})]
                )
                train_hitrate_results = pd.concat(
                    [
                        train_hitrate_results,
                        pd.DataFrame({"offset": [offset], "hitrate": [train_hitrate]}),
                    ]
                )

                # --------------------------------------------------------------
                # Evaluation on test set
                # --------------------------------------------------------------

                test_mse = mean_squared_error(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_r2 = r2_score(y_test, y_test_pred)

                test_hitrate = (
                    np.sign(pd.Series(y_test_pred, name="y_test_pred").diff())
                    == np.sign(y_test.diff())
                ).mean()

                # Append test metrics
                test_mse_results = pd.concat(
                    [test_mse_results, pd.DataFrame({"offset": [offset], "mse": [test_mse]})]
                )
                test_mae_results = pd.concat(
                    [test_mae_results, pd.DataFrame({"offset": [offset], "mae": [test_mae]})]
                )
                test_r2_results = pd.concat(
                    [test_r2_results, pd.DataFrame({"offset": [offset], "r2": [test_r2]})]
                )
                test_hitrate_results = pd.concat(
                    [
                        test_hitrate_results,
                        pd.DataFrame({"offset": [offset], "hitrate": [test_hitrate]}),
                    ]
                )

            # --------------------------------------------------------------
            # Save aggregated metrics for the current target
            # --------------------------------------------------------------

            train_mse_results.to_csv(target_path / "rev_train_mse_results.csv", index=False)
            train_mae_results.to_csv(target_path / "rev_train_mae_results.csv", index=False)
            train_r2_results.to_csv(target_path / "rev_train_r2_results.csv", index=False)
            train_hitrate_results.to_csv(target_path / "rev_train_hitrate_results.csv", index=False)

            test_mse_results.to_csv(target_path / "rev_test_mse_results.csv", index=False)
            test_mae_results.to_csv(target_path / "rev_test_mae_results.csv", index=False)
            test_r2_results.to_csv(target_path / "rev_test_r2_results.csv", index=False)
            test_hitrate_results.to_csv(target_path / "rev_test_hitrate_results.csv", index=False)

