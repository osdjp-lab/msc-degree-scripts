#!/usr/bin/env python3

"""Train ARIMA models with Optuna hyperparameter optimization on dataset/target/offset combinations."""

import os
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

INPUT_DIR = Path("../../data/1-preprocessing/7-split")
OUTPUT_DIR = Path("../../data/2-training-testing/arima")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

def arima_objective(trial, y_train):
    """Optuna objective: minimize MSE for given ARIMA(p,d,q)."""
    
    p = trial.suggest_int("p", 0, 6)
    d = trial.suggest_int("d", 0, 6) 
    q = trial.suggest_int("q", 0, 6)
    
    try:
        # Fit ARIMA model
        model = ARIMA(y_train, order=(p, d, q))
        fitted_model = model.fit()
        
        # In-sample prediction
        y_pred = fitted_model.fittedvalues
        
        # CV-like score using last 30% of train as validation
        split = int(len(y_train) * 0.7)
        mse = mean_squared_error(y_train[split:], y_pred[split:])
        
        return mse
        
    except:
        return float('inf')  # Bad model → high score

for dataset_type in os.listdir(INPUT_DIR):
    if not (INPUT_DIR / dataset_type).is_dir():
        continue
        
    # Only stationary datasets
    valid_types = ['differenced', 'log-differenced', 'diff-standardized', 
                   'log-diff-standardized', 'diff-normalized', 'log-diff-normalized']
    if dataset_type not in valid_types:
        continue
        
    dataset_path = INPUT_DIR / dataset_type

    for target in os.listdir(dataset_path):
        target_path = dataset_path / target

        # Results containers
        train_results = {metric: [] for metric in ["mse", "mae", "r2", "hitrate"]}
        test_results = {metric: [] for metric in ["mse", "mae", "r2", "hitrate"]}

        for offset in os.listdir(target_path):
            print(f"{dataset_type}/{target}/{offset}")
            
            source_dir = target_path / offset
            result_dir = OUTPUT_DIR / dataset_type / target / offset
            result_dir.mkdir(parents=True, exist_ok=True)

            # Load data
            train_data = pd.read_csv(source_dir / "train_data.csv")
            test_data = pd.read_csv(source_dir / "test_data.csv")

            train_date = train_data.iloc[:, 0]
            y_train = train_data.iloc[:, -1]
            test_date = test_data.iloc[:, 0]
            y_test = test_data.iloc[:, -1]

            # Optuna optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda trial: arima_objective(trial, y_train), n_trials=1000)
            
            best_params = study.best_params
            best_params_path = result_dir / "best_params.json"
            
            with best_params_path.open("w") as f:
                json.dump({
                    "mse": study.best_value,
                    "params": best_params,
                    "n_trials": len(study.trials)
                }, f, indent=2)

            print(f"Best ARIMA({best_params['p']},{best_params['d']},{best_params['q']})")
            print(f"  Best MSE: {study.best_value:.6f}")

            # Fit final model with best params
            final_model = ARIMA(y_train, order=(best_params['p'], best_params['d'], best_params['q']))
            final_fitted = final_model.fit()

            # Train predictions
            y_train_pred = final_fitted.fittedvalues
            
            # Test predictions (offset ahead like your RF)
            offset_num = int(offset)
            y_test_pred_full = final_fitted.forecast(steps=len(y_test) + offset_num)
            y_test_pred = y_test_pred_full[offset_num:offset_num + len(y_test)]

            # Metrics (matching your RF exactly)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            train_hitrate = (np.sign(y_train_pred.values) == np.sign(y_train.values)).mean()

            test_mse = mean_squared_error(y_test.values, y_test_pred)
            test_mae = mean_absolute_error(y_test.values, y_test_pred)
            test_r2 = r2_score(y_test.values, y_test_pred)
            test_hitrate = (np.sign(y_test_pred) == np.sign(y_test.values)).mean()

            # Store results
            for results, metrics in [(train_results, [train_mse, train_mae, train_r2, train_hitrate]),
                                   (test_results, [test_mse, test_mae, test_r2, test_hitrate])]:
                results["mse"].append(test_mse)
                results["mae"].append(test_mae) 
                results["r2"].append(test_r2)
                results["hitrate"].append(test_hitrate)

            # Save predictions (bulletproof version)
            test_out = pd.DataFrame({
                'Date': test_date.values,
                'y_test': y_test.values,
                'y_test_pred': y_test_pred
            })
            test_out.to_csv(result_dir / "test_pred.csv", index=False)

            train_out = pd.DataFrame({
                'Date': train_date.values,
                'y_train': y_train.values,
                'y_train_pred': y_train_pred
            })
            train_out.to_csv(result_dir / "train_pred.csv", index=False)

        # Save aggregated results
        agg_path = OUTPUT_DIR / dataset_type / target
        agg_path.mkdir(parents=True, exist_ok=True)
        
        for metric, values in train_results.items():
            pd.DataFrame({"offset": list(os.listdir(target_path)), metric: values}).to_csv(
                agg_path / f"train_{metric}_results.csv", index=False)
        
        for metric, values in test_results.items():
            pd.DataFrame({"offset": list(os.listdir(target_path)), metric: values}).to_csv(
                agg_path / f"test_{metric}_results.csv", index=False)

        print(f"{target} complete")

print("ARIMA Optuna optimization complete!")

