import pytest
from unittest.mock import patch, MagicMock
import src.visualise_predictions as vis_pred

def dummy_preds(*args, **kwargs):
    import pandas as pd
    return pd.DataFrame({
        'id': ['A', 'A', 'B', 'B'],
        'date': ['2016-03-28', '2016-03-29', '2016-03-28', '2016-03-29'],
        'pred': [1, 2, 3, 4]
    })

def dummy_actuals(*args, **kwargs):
    import pandas as pd
    return pd.DataFrame({
        'id': ['A', 'B'],
        'item_id': ['A', 'B'],
        'dept_id': ['A', 'B'],
        'cat_id': ['A', 'B'],
        'store_id': ['CA_1', 'CA_1'],
        'state_id': ['X', 'Y'],
        'd_1': [1, 3],
        'd_2': [2, 4]
    })

def dummy_calendar(*args, **kwargs):
    import pandas as pd
    return pd.DataFrame({'date':['2016-03-28','2016-03-29'], 'd':['d_1','d_2']})

def parquet_side_effect(path, *args, **kwargs):
    path_str = str(path)
    if 'calendar' in path_str:
        return dummy_calendar()
    elif 'val_preds' in path_str:
        return dummy_preds()
    else:
        return dummy_actuals()

@patch('src.visualise_predictions.pd.read_parquet', side_effect=parquet_side_effect)
def test_plot_store_preds_vs_actuals(mock_read):
    fig = vis_pred.plot_store_preds_vs_actuals('CA_1')
    assert fig is None or hasattr(fig, 'to_html')

@patch('src.visualise_predictions.pd.read_parquet', side_effect=parquet_side_effect)
def test_main_runs(mock_read):
    with patch('builtins.open', new_callable=MagicMock()):
        vis_pred.main()
