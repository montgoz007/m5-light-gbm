"""
visualise_predictions.py

Generates an HTML report comparing actuals and predictions for each store in the M5 Forecasting dataset.
"""
import pandas as pd
import plotly.graph_objs as go
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data/m5-forecasting-accuracy/processed"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
TRAIN_PATH = DATA_DIR / "sales_train_validation.parquet"

STORES = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3"
]

REPORT_PATH = OUTPUT_DIR / "predictions_vs_actuals_report.html"


def plot_store_preds_vs_actuals(store_id):
    preds_path = OUTPUT_DIR / f"{store_id}_val_preds.parquet"
    if not preds_path.exists():
        print(f"Prediction file not found for {store_id}: {preds_path}")
        return None
    preds = pd.read_parquet(preds_path)
    if not set(['id', 'date', 'pred']).issubset(preds.columns):
        print(f"Predictions file for {store_id} is not in expected long format (id, date, pred). Columns: {preds.columns.tolist()}")
        return None
    # Load calendar to map date to d_x
    calendar = pd.read_parquet(DATA_DIR / "calendar.parquet")
    date_map = dict(zip(calendar['date'].astype(str), calendar['d']))
    preds['date'] = preds['date'].astype(str).map(date_map)
    # Get actuals from validation set (wide format)
    val = pd.read_parquet(TRAIN_PATH)
    val = val[val['store_id'] == store_id]
    value_vars = [c for c in val.columns if c not in ['id','item_id','dept_id','cat_id','store_id','state_id']]
    val_long = val.melt(id_vars=['id'], value_vars=value_vars, var_name='date', value_name='actual')
    preds['date'] = preds['date'].astype(str)
    val_long['date'] = val_long['date'].astype(str)
    merged = preds.merge(val_long, on=['id', 'date'])
    if merged.empty:
        print(f"No matching rows after merging predictions and actuals for {store_id}.")
        return None
    # Aggregate total actuals and predictions per date
    agg = merged.groupby('date').agg({'actual':'sum', 'pred':'sum'}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg['date'], y=agg['actual'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=agg['date'], y=agg['pred'], mode='lines', name='Predicted'))
    fig.update_layout(title=f"Total Sales: Actual vs Predicted for {store_id}", xaxis_title="Day", yaxis_title="Total Sales")
    return fig

def main():
    figs = []
    for store in STORES:
        fig = plot_store_preds_vs_actuals(store)
        if fig:
            figs.append(fig)
    with open(REPORT_PATH, "w") as f:
        for fig in figs:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    print(f"Predictions vs Actuals report saved to {REPORT_PATH.resolve()}")

if __name__ == "__main__":
    main()
