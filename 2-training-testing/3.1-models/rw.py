#!/usr/bin/env python3

"""Generate Random Walk forecasts - FIXED SAVING."""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_DIR = Path("../../data/1-preprocessing/7-split")
OUTPUT_DIR = Path("../../data/2-training-testing/rw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def random_walk_forecast(y_train, offset_num):
    """RW: repeat last train value."""
    return np.full(len(y_train), y_train.iloc[-1])

def random_walk_simulation(y_train, n_steps):
    """Single RW simulation with white noise (matches point forecast shape)."""
    last_value = y_train.iloc[-1]
    noise_std = y_train.diff().std()  # σ_ε from train differences
    
    # Single simulation path
    noise = np.random.normal(0, noise_std, n_steps)
    rw_path = last_value + np.cumsum(noise)
    
    return rw_path

for dataset_type in os.listdir(INPUT_DIR):
    if not (INPUT_DIR / dataset_type).is_dir():
        continue
        
    if "raw" not in dataset_type:
        continue
        
    dataset_path = INPUT_DIR / dataset_type
    for target in os.listdir(dataset_path):
        target_path = dataset_path / target

        # FIXED: Initialize as lists, convert to DF at end
        train_metrics = {"mse": [], "mae": [], "r2": [], "hitrate": []}
        test_metrics = {"mse": [], "mae": [], "r2": [], "hitrate": []}
        offsets = []

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

            offset_num = int(offset)

            # RW FORECASTS
            last_train = y_train.iloc[-1]
            y_train_pred = random_walk_simulation(y_train, len(y_train))
            y_test_pred = random_walk_simulation(y_test, len(y_test))
            # y_train_pred = np.full(len(y_train), last_train)
            # y_test_pred = np.full(len(y_test), last_train)

            # FIXED METRICS CALCULATION
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            if "diff" in dataset_type:
                train_hitrate = (np.sign(y_train_pred) == np.sign(y_train)).mean()
            else:
                train_hitrate = (
                    np.sign(pd.Series(y_train_pred).diff()) == np.sign(y_train.diff())
                ).mean()

            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            if "diff" in dataset_type:
                test_hitrate = (np.sign(y_test_pred) == np.sign(y_test)).mean()
            else:
                test_hitrate = (
                    np.sign(pd.Series(y_test_pred).diff()) == np.sign(y_test.diff())
                ).mean()

            offsets.append(offset)
            train_metrics["mse"].append(train_mse)
            train_metrics["mae"].append(train_mae)
            train_metrics["r2"].append(train_r2)
            train_metrics["hitrate"].append(train_hitrate)
            
            test_metrics["mse"].append(test_mse)
            test_metrics["mae"].append(test_mae)
            test_metrics["r2"].append(test_r2)
            test_metrics["hitrate"].append(test_hitrate)

            train_out = pd.DataFrame({
                'Date': train_date.values,
                'y_train': y_train.values,
                'y_train_pred': y_train_pred
            })
            train_out.to_csv(result_dir / "train_pred.csv", index=False)
            print(f"Saved {result_dir / 'train_pred.csv'}")

            test_out = pd.DataFrame({
                'Date': test_date.values,
                'y_test': y_test.values,
                'y_test_pred': y_test_pred
            })
            test_out.to_csv(result_dir / "test_pred.csv", index=False)
            print(f"Saved {result_dir / 'test_pred.csv'}")

            # RW info
            with (result_dir / "rw_info.json").open("w") as f:
                json.dump({"last_train": float(last_train), "forecast": float(last_train)}, f)

        # FIXED: Save aggregated metrics at END
        agg_path = OUTPUT_DIR / dataset_type / target
        agg_path.mkdir(parents=True, exist_ok=True)
        
        # Train metrics
        for metric in ["mse", "mae", "r2", "hitrate"]:
            pd.DataFrame({
                "offset": offsets,
                metric: train_metrics[metric]
            }).to_csv(agg_path / f"train_{metric}_results.csv", index=False)
            print(f"Saved {agg_path / f'train_{metric}_results.csv'}")
        
        # Test metrics  
        for metric in ["mse", "mae", "r2", "hitrate"]:
            pd.DataFrame({
                "offset": offsets,
                metric: test_metrics[metric]
            }).to_csv(agg_path / f"test_{metric}_results.csv", index=False)
            print(f"Saved {agg_path / f'test_{metric}_results.csv'}")

        print(f"{target} complete")

print("Random Walk complete!")

