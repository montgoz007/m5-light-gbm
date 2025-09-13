import pytest
import pandas as pd
import numpy as np
from src import prepare_data

def test_create_features_basic():
    # Minimal DataFrame with required columns
    df = pd.DataFrame({
        'id': ['A', 'A', 'A', 'B', 'B', 'B'],
        'sales': [1, 2, 3, 4, 5, 6],
        'date': pd.date_range('2020-01-01', periods=6),
        'event_name_1': ['a']*6,
        'event_type_1': ['b']*6,
        'event_name_2': ['c']*6,
        'event_type_2': ['d']*6,
    })
    out = prepare_data.create_features(df.copy())
    # Check new columns exist
    for col in ['year','month','week','day','dayofweek','lag_7','lag_28','rolling_mean_7','rolling_mean_28']:
        assert col in out.columns
    # Check types
    assert out['year'].dtype == np.int16
    assert out['month'].dtype == np.int8
    assert out['week'].dtype == np.int8
    assert out['day'].dtype == np.int8
    assert out['dayofweek'].dtype == np.int8
    # Check event columns are int16
    for col in ['event_name_1','event_type_1','event_name_2','event_type_2']:
        assert out[col].dtype == np.int16
    # Check lag columns are float32
    for col in ['lag_7','lag_28','rolling_mean_7','rolling_mean_28']:
        assert out[col].dtype == np.float32
