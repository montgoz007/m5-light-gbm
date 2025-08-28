#!/usr/bin/env python3
"""
train_model.py
Trains LightGBM models on M5 Forecasting features with dynamic last‑28‑day validation split.
"""

import pandas as pd
import gc
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
FEATURES_DIR = Path("data/processed_features")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FEATURES = [
    'year','month','week','day','dayofweek',
    'event_name_1','event_type_1','event_name_2','event_type_2',
    'sell_price','lag_7','lag_28','rolling_mean_7','rolling_mean_28'
]
TARGET = "sales"

# ---------------------------
# Load feature files
# ---------------------------
feature_files = sorted(FEATURES_DIR.glob("*_features.parquet"))
all_val_scores = []

for file in tqdm(feature_files, desc="Training stores"):
    df = pd.read_parquet(file)
    store_id = file.stem.replace("_features", "")

    if 'split' not in df.columns:
        continue

    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']

    if train_df.empty or val_df.empty:
        continue

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        train_df[FEATURES], train_df[TARGET],
        eval_set=[(val_df[FEATURES], val_df[TARGET])],
        eval_metric='rmse',
        #early_stopping_rounds=50
    )

    val_pred = model.predict(val_df[FEATURES])
    rmse = mean_squared_error(val_df[TARGET], val_pred)
    all_val_scores.append((store_id, rmse))

    preds_df = val_df[['id','date']].copy()
    preds_df['pred'] = val_pred
    preds_df.to_parquet(OUTPUT_DIR / f"{store_id}_val_preds.parquet", index=False)

    del df, train_df, val_df, preds_df, model
    gc.collect()

# ---------------------------
# Save summary
# ---------------------------
score_df = pd.DataFrame(all_val_scores, columns=['store_id','val_rmse'])
score_df.to_csv(OUTPUT_DIR / "validation_scores.csv", index=False)

print("\n=== VALIDATION SUMMARY ===")
print(f"Trained stores: {len(score_df)}")
if not score_df.empty:
    print(f"Average RMSE: {score_df['val_rmse'].mean():.4f}")