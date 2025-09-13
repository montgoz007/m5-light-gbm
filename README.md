# m5-light-gbm

LightGBM-based solution for the Kaggle M5 Forecasting - Accuracy competition. This project provides scripts for data acquisition, feature engineering, model training, and cross-validation for time-series forecasting at the store/item level.

## Features
- Automated download and processing of M5 competition data
- Feature engineering with lag and rolling statistics
- Per-store LightGBM model training with dynamic validation split
- Time-series cross-validation and hyperparameter search
- Outputs validation scores and best parameters per store
- Modular, testable codebase with CLI utilities

## Project Structure

```
├── data/
│   └── m5-forecasting-accuracy/
│       ├── raw/         # Raw CSVs from Kaggle
│       ├── processed/   # Parquet-processed data
│       └── processed_features/ # Per-store feature files
├── output/              # Model outputs, predictions, validation scores
├── src/                 # Source code
│   ├── get_kaggle_data.py   # Download and process data
│   ├── prepare_data.py      # Feature engineering
│   ├── train_model.py       # Train LightGBM models
│   └── cross_validate.py    # Cross-validation and hyperparameter search
├── tests/               # Unit tests
├── requirements.txt     # Python dependencies
├── pytest.ini           # Pytest config
└── README.md
```

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/montgoz007/m5-light-gbm.git
	cd m5-light-gbm
	```
2. Install dependencies (ideally in a virtual environment):
	```bash
	pip install -r requirements.txt
	```
3. Set up Kaggle API credentials as environment variables or in a `.env` file:
	```env
	KAGGLE_USERNAME=your_username
	KAGGLE_KEY=your_key
	```

## Usage

### 1. Download and Prepare Data

```bash
python src/get_kaggle_data.py m5-forecasting-accuracy
```

### 2. Feature Engineering

```bash
python src/prepare_data.py
```

### 3. Train Models

```bash
python src/train_model.py
```

### 4. Cross-Validation & Hyperparameter Search

```bash
python src/cross_validate.py --search grid  # or --search random
```


## Visualisation

### Exploratory Data Analysis (EDA)
Generate an interactive HTML report to explore the M5 dataset before modeling:

```bash
python src/visualise_data.py
```
Output: `output/eda_report.html` (includes sales trends, per-store breakdowns, and more)

### Predictions vs Actuals
Visualise model predictions against actuals for each store:

```bash
python src/visualise_predictions.py
```
Output: `output/predictions_vs_actuals_report.html` (charts for each store, overlaying predictions and actuals)

## Outputs
- Per-store validation predictions: `output/*_val_preds.parquet`
- Validation scores summary: `output/validation_scores.csv`
- Best parameters per store: `output/best_params.json`
- EDA report: `output/eda_report.html`
- Predictions vs Actuals report: `output/predictions_vs_actuals_report.html`

## Testing

Run all unit tests:
```bash
pytest
```

## Requirements

See `requirements.txt` for all dependencies. Main packages:
- Python 3.8+
- pandas, numpy, scikit-learn, lightgbm, typer, kaggle, pyarrow, tqdm

## License

MIT License. See [LICENSE](LICENSE) if present.

## Acknowledgements
- [Kaggle M5 Forecasting - Accuracy Competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- LightGBM, pandas, scikit-learn, and the open-source Python community