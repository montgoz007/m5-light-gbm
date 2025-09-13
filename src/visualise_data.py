"""
visualize_data.py

Generates an HTML report with exploratory data analysis (EDA) charts for the M5 Forecasting dataset using Plotly.
Run this script before modeling to understand the data.
"""

import os
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
from pathlib import Path



# Robust paths based on script location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data/m5-forecasting-accuracy/processed"
TRAIN_PATH = DATA_DIR / "sales_train_validation.parquet"
EVAL_PATH = DATA_DIR / "sales_train_evaluation.parquet"
CALENDAR_PATH = DATA_DIR / "calendar.parquet"
PRICES_PATH = DATA_DIR / "sell_prices.parquet"

OUTPUT_REPORT = SCRIPT_DIR.parent / "output/eda_report.html"

def load_data():
    train = pd.read_parquet(TRAIN_PATH)
    eval_ = pd.read_parquet(EVAL_PATH)
    calendar = pd.read_parquet(CALENDAR_PATH)
    prices = pd.read_parquet(PRICES_PATH)
    return train, eval_, calendar, prices

def plot_sales_over_time(train):
    # Aggregate sales by day
    sales_by_day = train.drop(['id','item_id','dept_id','cat_id','store_id','state_id'], axis=1).sum(axis=0)
    fig = px.line(x=sales_by_day.index, y=sales_by_day.values, labels={'x':'Day','y':'Total Sales'}, title='Total Sales Over Time (Train)')
    return fig

def plot_sample_series(train, n=5):
    # Plot a few random series
    sample = train.sample(n)
    fig = sp.make_subplots(rows=n, cols=1, shared_xaxes=True, subplot_titles=sample['id'].tolist())
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        fig.add_trace(go.Scatter(x=train.columns[6:], y=row[6:], mode='lines', name=row['id']), row=i, col=1)
    fig.update_layout(height=200*n, title_text="Sample Item Sales Series (Train)")
    return fig

def plot_sales_distribution(train):
    # Distribution of total sales per item
    item_totals = train.iloc[:,6:].sum(axis=1)
    fig = px.histogram(item_totals, nbins=50, title="Distribution of Total Sales per Item (Train)", labels={'value':'Total Sales'})
    return fig



def plot_store_sales_over_time(train):
    # Plot total sales over time for each store
    stores = train['store_id'].unique()
    fig = go.Figure()
    for store in stores:
        store_sales = train[train['store_id'] == store].drop(['id','item_id','dept_id','cat_id','store_id','state_id'], axis=1).sum(axis=0)
        fig.add_trace(go.Scatter(x=store_sales.index, y=store_sales.values, mode='lines', name=store))
    fig.update_layout(title="Total Sales Over Time by Store (Train)", xaxis_title="Day", yaxis_title="Total Sales")
    return fig


def main():
    train, eval_, calendar, prices = load_data()
    figs = []
    figs.append(plot_sales_over_time(train))
    figs.append(plot_sample_series(train))
    figs.append(plot_sales_distribution(train))
    figs.append(plot_store_sales_over_time(train))
    # Save all figures to a single HTML file
    with open(OUTPUT_REPORT, "w") as f:
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    print(f"EDA report saved to {OUTPUT_REPORT.resolve()}")

if __name__ == "__main__":
    main()
