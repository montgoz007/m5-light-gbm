#!/usr/bin/env python3
"""
cross_validate.py
Standalone time-series cross-validation for M5 per-store feature sets.
Performs walk-forward CV and saves best LightGBM params per store to JSON.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid, ParameterSampler
from tqdm import tqdm
import gc
import json
import argparse

# ---------------------------
# Config
# ---------------------------
FEATURES_DIR = Path("data/processed_features")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

BEST_PARAMS_FILE = OUTPUT_DIR / "best_params.json"

FEATURES = [
    'year','month','week','day','dayofweek',
    'event_name_1','event_type_1','event_name_2','event_type_2',
    'sell_price','lag_7','lag_28','rolling_mean_7','rolling_mean_28'
]
TARGET = "sales"

# Hyperparameter search space (keep small for speed)
PARAM_GRID = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'num_leaves': [5, 10, 20, 40],
    'min_child_samples': [5, 10, 20, 50]
}

FOLDS = 3            # Number of CV folds
VAL_SIZE_DAYS = 28   # Size of each validation block in days

# ---------------------------
# Cross-validation function
# ---------------------------
def time_series_cv(df, params_grid, folds, val_days, search_type="grid", n_iter=10, random_state=42):
    """Performs walk-forward CV and returns best params + RMSE. Supports grid or randomized search."""
    dates_sorted = df['date'].sort_values().unique()
    results = []

    if search_type == "random":
        param_iter = ParameterSampler(params_grid, n_iter=n_iter, random_state=random_state)
    else:
        param_iter = ParameterGrid(params_grid)

    for params in param_iter:
        fold_rmses = []

        # Walk forward through the last `folds` validation blocks
        for f in range(folds, 0, -1):
            val_start = dates_sorted[-(val_days * f)]
            val_end   = dates_sorted[-(val_days * (f - 1))] if f > 1 else dates_sorted[-1]

            fold_train = df[df['date'] < val_start]
            fold_val   = df[(df['date'] >= val_start) & (df['date'] <= val_end)]

            if fold_train.empty or fold_val.empty:
                continue

            model = LGBMRegressor(
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **params
            )
            model.fit(fold_train[FEATURES], fold_train[TARGET])
            preds = model.predict(fold_val[FEATURES])
            rmse = mean_squared_error(fold_val[TARGET], preds) ** 0.5
            fold_rmses.append(rmse)

        if fold_rmses:
            avg_rmse = np.mean(fold_rmses)
            results.append((params, avg_rmse))

    return sorted(results, key=lambda x: x[1])[0] if results else (None, None)

# ---------------------------
# Main loop
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Time-series cross-validation for M5 LightGBM models.")
    parser.add_argument('--search', choices=['grid', 'random'], default='grid', help='Search type: grid or random (default: grid)')
    parser.add_argument('--n_iter', type=int, default=10, help='Number of random samples for random search (default: 10)')
    args = parser.parse_args()

    feature_files = sorted(FEATURES_DIR.glob("*_features.parquet"))
    best_params_all = {}

    for file in tqdm(feature_files, desc="Cross-validating stores"):
        df = pd.read_parquet(file)
        store_id = file.stem.replace("_features", "")

        if 'split' in df.columns:
            # Use all data for CV
            df = df.drop(columns=['split'])

        df = df.sort_values(['id','date'])

        best_params, best_rmse = time_series_cv(
            df, PARAM_GRID, FOLDS, VAL_SIZE_DAYS,
            search_type=args.search, n_iter=args.n_iter
        )

        if best_params:
            best_params_all[store_id] = {
                "best_params": best_params,
                "avg_rmse": best_rmse
            }
            print(f"{store_id} → Best params: {best_params}, Avg RMSE: {best_rmse:.4f}")
        else:
            print(f"{store_id} → No valid folds")

        del df
        gc.collect()

    # Save all best params to JSON
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params_all, f, indent=2)

    print(f"\n✅ Best params saved to {BEST_PARAMS_FILE}")

if __name__ == "__main__":
    main()