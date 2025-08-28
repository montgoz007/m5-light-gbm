#!/usr/bin/env python3
"""
prepare_data.py
Processes M5 Forecasting data in memory‑safe batches.
Splits dynamically: last 28 days of available history = validation.
Ensures both train and val rows exist for every store.
"""

import pandas as pd
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
DATA_DIR = Path("data/m5-forecasting-accuracy/processed")
FEATURES_DIR = Path("data/processed_features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LAGS = [7, 28]
DEFAULT_WINDOWS = [7, 28]

# ---------------------------
# Load static data
# ---------------------------
print("Loading static datasets...")
calendar = pd.read_parquet(DATA_DIR / "calendar.parquet")
prices   = pd.read_parquet(DATA_DIR / "sell_prices.parquet")
sales    = pd.read_parquet(DATA_DIR / "sales_train_validation.parquet")

for col in ['event_name_1','event_type_1','event_name_2','event_type_2']:
    calendar[col] = calendar[col].astype('category')

# ---------------------------
# Feature engineering
# ---------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year.astype('int16')
    df['month'] = df['date'].dt.month.astype('int8')
    df['week'] = df['date'].dt.isocalendar().week.astype('int8')
    df['day'] = df['date'].dt.day.astype('int8')
    df['dayofweek'] = df['date'].dt.dayofweek.astype('int8')

    for col in ['event_name_1','event_type_1','event_name_2','event_type_2']:
        df[col] = df[col].astype('category').cat.codes.astype('int16')

    for lag in DEFAULT_LAGS:
        df[f'lag_{lag}'] = df.groupby('id')['sales'].shift(lag).astype('float32')
    for window in DEFAULT_WINDOWS:
        df[f'rolling_mean_{window}'] = (
            df.groupby('id')['sales'].shift(1).rolling(window).mean().astype('float32')
        )
    return df

# ---------------------------
# Process per store
# ---------------------------
stores = sales['store_id'].unique()
print(f"Processing {len(stores)} stores with dynamic last‑28‑day split...")

for store in tqdm(stores, desc="Stores"):
    subset_sales = sales[sales['store_id'] == store].copy()

    id_vars = ['id','item_id','dept_id','cat_id','store_id','state_id']
    melted = subset_sales.melt(id_vars=id_vars, var_name='d_raw', value_name='sales')
    melted['d_raw'] = melted['d_raw'].astype(str)

    # Merge calendar (so we have actual dates before split)
    cal_renamed = calendar.rename(columns={'d': 'd_raw'})
    merged = melted.merge(cal_renamed, on='d_raw', how='left')

    # Merge prices
    merged = merged.merge(
        prices[prices['store_id'] == store],
        on=['store_id','item_id','wm_yr_wk'], how='left'
    )

    # Feature engineering
    merged = create_features(merged)

    # Drop rows where all lags are missing
    merged.dropna(subset=[f'lag_{l}' for l in DEFAULT_LAGS], how='all', inplace=True)

    # Assign split: last 28 days in *this store's available history*
    merged = merged.sort_values(['id','date'])
    max_date = merged['date'].max()
    val_start_date = max_date - pd.Timedelta(days=27)  # inclusive of max_date

    merged['split'] = np.where(merged['date'] < val_start_date, 'train', 'val')

    # Save per‑store feature file
    merged.to_parquet(FEATURES_DIR / f"{store}_features.parquet", index=False)

    del subset_sales, melted, merged
    gc.collect()

print("✅ Feature engineering complete — dynamic train/val split applied for all stores")