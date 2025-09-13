import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

import src.visualise_data as vis_data

def dummy_df(*args, **kwargs):
    # Return a small DataFrame with the right columns for all functions
    cols = ['id','item_id','dept_id','cat_id','store_id','state_id'] + [f'd_{i}' for i in range(1, 4)]
    data = [['A','A','A','A','S1','X',1,2,3], ['B','B','B','B','S2','Y',4,5,6]]
    return pd.DataFrame(data, columns=cols)

def dummy_calendar(*args, **kwargs):
    return pd.DataFrame({'date':['2016-03-28','2016-03-29','2016-03-30'], 'd':['d_1','d_2','d_3']})

def dummy_prices(*args, **kwargs):
    return pd.DataFrame({'dummy':[1,2,3]})

@patch('src.visualise_data.pd.read_parquet', side_effect=[dummy_df(), dummy_df(), dummy_calendar(), dummy_prices()])
def test_load_data(mock_read):
    train, eval_, calendar, prices = vis_data.load_data()
    assert train.shape[0] > 0
    assert eval_.shape[0] > 0
    assert calendar.shape[0] > 0
    assert prices.shape[0] > 0

@patch('src.visualise_data.pd.read_parquet', return_value=dummy_df())
def test_plot_sales_over_time(mock_read):
    fig = vis_data.plot_sales_over_time(dummy_df())
    assert fig is not None
    assert hasattr(fig, 'to_html')

@patch('src.visualise_data.pd.read_parquet', return_value=dummy_df())
def test_plot_sample_series(mock_read):
    fig = vis_data.plot_sample_series(dummy_df(), n=2)
    assert fig is not None
    assert hasattr(fig, 'to_html')

@patch('src.visualise_data.pd.read_parquet', return_value=dummy_df())
def test_plot_sales_distribution(mock_read):
    fig = vis_data.plot_sales_distribution(dummy_df())
    assert fig is not None
    assert hasattr(fig, 'to_html')

@patch('src.visualise_data.pd.read_parquet', return_value=dummy_df())
def test_plot_store_sales_over_time(mock_read):
    fig = vis_data.plot_store_sales_over_time(dummy_df())
    assert fig is not None
    assert hasattr(fig, 'to_html')
