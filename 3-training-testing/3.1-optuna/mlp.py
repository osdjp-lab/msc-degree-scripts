#!/usr/bin/env python3

"""Train an MLPRegressor on many dataset/target/offset combinations
with hyper‑parameter optimisation via OptunaSearchCV."""

import os
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor

INPUT_DIR = Path("../../data/1-preprocessing/7-split")
OUTPUT_DIR = Path("../../data/3-training-testing/optuna/mlp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_search_space() -> dict:
    """Return the dictionary that OptunaSearchCV expects."""
    return {
        # v1
        "hidden_layer_sizes": optuna.distributions.IntDistribution(low=1, high=30),
        #"solver": optuna.distributions.CategoricalDistribution(['adam', 'sgd', 'lbfgs']),
        "solver": optuna.distributions.CategoricalDistribution(['adam']),
        "alpha": optuna.distributions.FloatDistribution(low=1e-10, high=1e-1, log=True),
        "tol": optuna.distributions.FloatDistribution(low=1e-8, high=1e-1, log=True),

        # v2
        # "hidden_layer_sizes": optuna.distributions.IntDistribution(low=1, high=10),
        # "solver": optuna.distributions.CategoricalDistribution(['adam']),
        # "alpha": optuna.distributions.FloatDistribution(low=1e-3, high=1e-1, log=True),
        # "tol": optuna.distributions.FloatDistribution(low=1e-3, high=1e-1, log=True),
    }

for dataset_type in os.listdir(INPUT_DIR):
    if "normalized" not in dataset_type:
        continue

    dataset_path = INPUT_DIR / dataset_type

    for target in os.listdir(dataset_path):
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
            print(f"{dataset_type}/{target}/{offset}")

            source_dir = target_path / offset
            result_dir = OUTPUT_DIR / dataset_type / target / offset
            result_dir.mkdir(parents=True, exist_ok=True)

            # --------------------------------------------------------------
            # Load data
            # --------------------------------------------------------------
            train_data = pd.read_csv(source_dir / "train_data.csv")
            test_data = pd.read_csv(source_dir / "test_data.csv")

            train_date = train_data.iloc[:, 0]
            X_train = train_data.iloc[:, 1:-1]
            y_train = train_data.iloc[:, -1]

            test_date = test_data.iloc[:, 0]
            X_test = test_data.iloc[:, 1:-1]
            y_test = test_data.iloc[:, -1]

            # --------------------------------------------------------------
            # Model + Optuna optimisation
            # --------------------------------------------------------------
            base_model = MLPRegressor(
                activation="tanh",
                shuffle=False,
                random_state=0,
                max_iter=10000,
            )

            optuna_search = optuna.integration.OptunaSearchCV(
                estimator=base_model,
                param_distributions=get_search_space(),
                n_trials=100,
                scoring="neg_mean_squared_error",
                cv=5,
                verbose=2,
                n_jobs=-1,
            )
            optuna_search.fit(X_train, y_train)

            # --------------------------------------------------------------
            # Save best‑trial information
            # --------------------------------------------------------------
            best_trial = optuna_search.study_.best_trial

            best_params_path = result_dir / "best_params.json"
            with best_params_path.open("w") as f:
                json.dump(
                    {
                        "value": best_trial.value,
                        "params": best_trial.params,
                    },
                    f,
                    indent=2,
                )
            # (optional) CSV version if needed
            # pd.DataFrame([best_trial.params]).to_csv(result_dir / "best_params.csv", index=False)

            print("Best trial:")
            print(f"  Value: {best_trial.value}")
            for k, v in best_trial.params.items():
                print(f"    {k}: {v}")

            # --------------------------------------------------------------
            # Evaluation on training set
            # --------------------------------------------------------------
            y_train_pred = optuna_search.predict(X_train)

            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)

            if "diff" in dataset_type:
                train_hitrate = (np.sign(y_train_pred) == np.sign(y_train)).mean()
            else:
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
            y_test_pred = optuna_search.predict(X_test)

            test_mse = mean_squared_error(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            if "diff" in dataset_type:
                test_hitrate = (np.sign(y_test_pred) == np.sign(y_test)).mean()
            else:
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
            # Save forecasts
            # --------------------------------------------------------------
            train_out = pd.concat(
                [
                    train_date,
                    pd.Series(y_train, name="y_train"),
                    pd.Series(y_train_pred, name="y_train_pred"),
                ],
                axis=1,
            )
            train_out.to_csv(result_dir / "train_pred.csv", index=False)

            test_out = pd.concat(
                [
                    test_date,
                    pd.Series(y_test, name="y_test"),
                    pd.Series(y_test_pred, name="y_test_pred"),
                ],
                axis=1,
            )
            test_out.to_csv(result_dir / "test_pred.csv", index=False)

        # --------------------------------------------------------------
        # Save aggregated metrics for the current target
        # --------------------------------------------------------------
        agg_path = OUTPUT_DIR / dataset_type / target
        agg_path.mkdir(parents=True, exist_ok=True)

        train_mse_results.to_csv(agg_path / "train_mse_results.csv", index=False)
        train_mae_results.to_csv(agg_path / "train_mae_results.csv", index=False)
        train_r2_results.to_csv(agg_path / "train_r2_results.csv", index=False)
        train_hitrate_results.to_csv(agg_path / "train_hitrate_results.csv", index=False)

        test_mse_results.to_csv(agg_path / "test_mse_results.csv", index=False)
        test_mae_results.to_csv(agg_path / "test_mae_results.csv", index=False)
        test_r2_results.to_csv(agg_path / "test_r2_results.csv", index=False)
        test_hitrate_results.to_csv(agg_path / "test_hitrate_results.csv", index=False)

        print("Set complete")

print("Calculations complete")
