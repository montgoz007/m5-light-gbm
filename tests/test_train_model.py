import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src import train_model

def make_df():
    # Minimal DataFrame with required columns and splits
    n = 10
    df = pd.DataFrame({
        'id': ['A']*n + ['B']*n,
        'date': pd.date_range('2020-01-01', periods=n).tolist()*2,
        'sales': np.random.randint(0, 10, 2*n),
        'year': 2020,
        'month': 1,
        'week': 1,
        'day': 1,
        'dayofweek': 1,
        'event_name_1': 0,
        'event_type_1': 0,
        'event_name_2': 0,
        'event_type_2': 0,
        'sell_price': 1.0,
        'lag_7': 1.0,
        'lag_28': 1.0,
        'rolling_mean_7': 1.0,
        'rolling_mean_28': 1.0,
        'split': ['train']*5 + ['val']*5 + ['train']*5 + ['val']*5
    })
    return df

@patch('src.train_model.tqdm', lambda x, **kwargs: x)  # disables tqdm progress bar
@patch('src.train_model.LGBMRegressor')
@patch('src.train_model.pd.read_parquet')
@patch('src.train_model.pd.DataFrame.to_parquet')
@patch('src.train_model.pd.DataFrame.to_csv')
@patch('src.train_model.Path.glob')
def test_main_runs(mock_glob, mock_to_csv, mock_to_parquet, mock_read_parquet, MockLGBM):
    # Mock feature files
    mock_file = MagicMock()
    mock_file.stem = 'A_features'
    mock_glob.return_value = [mock_file]
    # Mock DataFrame loading
    mock_read_parquet.return_value = make_df()
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda X: np.ones(len(X))
    MockLGBM.return_value = mock_model
    # Run main
    train_model.main()
    # Check that model was fit and predictions saved
    assert mock_model.fit.called
    assert mock_to_parquet.called
    assert mock_to_csv.called
