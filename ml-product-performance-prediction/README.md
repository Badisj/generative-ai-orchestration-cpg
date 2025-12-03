# ML Product Performance Prediction

Purpose
-------
This project estimates product target variables for chocolate formulations (sensory, physical, and stability targets) and includes tooling to train ML models that predict those targets from formulation and processing features.

Repository layout (high level)
--------------------------------
- `data/` — input CSVs. Example: `Chocolate_bar_dataset_with_features.csv`.
- `src/process/2. Chocolate_target_estimation.py` — contains `estimate_chocolate_targets(df)` and the canonical `target_columns` list.
- `src/train/train_targets.py` — CLI script to train Random Forest and Gradient Boosting models per target and save artifacts.
- `notebooks/` — notebooks for demo and analysis. Prefer `notebooks/Train_and_Evaluate_Targets_fixed.ipynb` for a robust executable demo.
- `outputs/` — trained models, metrics, and exported artifacts are written here.

Quick setup
-----------
Recommended: create a virtual environment and install dependencies.

PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train models (CLI)
------------------
The training script will:
- Read `target_columns` from `src/process/2. Chocolate_target_estimation.py`.
- Load your input CSV and prepare numeric features (drops columns in `target_columns`).
- Train a Random Forest and a Gradient Boosting model per target (classification for `shelf_life_exceeded`).

Example (PowerShell):

```powershell
python -m src.train.train_targets --input data/Chocolate_bar_dataset_with_features.csv --output-dir outputs/models --test-size 0.20 --random-state 42
```

Key CLI options:
- `--input` (required): path to input CSV.
- `--process-source`: path to the process file with `target_columns` (default: `src/process/2. Chocolate_target_estimation.py`).
- `--output-dir`: output directory for models and metrics (default: `outputs/models`).
- `--test-size`: test split fraction (default `0.2`).
- `--random-state`: seed for reproducibility (default `42`).

Outputs produced
----------------
- Model artifacts: Joblib files under `outputs/models/models/<target>__<model>.joblib`.
- Metrics summary: `outputs/models/metrics_summary.json` (per-target & per-model metrics and model paths).

Notebooks
---------
- `notebooks/Train_and_Evaluate_Targets_fixed.ipynb` is a ready-to-run demo that: loads data, runs the estimator if targets are missing, trains models, and saves metrics. Use it interactively or execute it programmatically.

Data expectations
-----------------
- Input is a CSV with one row per formulation.
- The training pipeline uses numeric columns as features. If you have categorical features, encode them (one-hot or ordinal) before running the script or extend the notebook/script to add preprocessing.

Modeling notes
--------------
- Regression targets use `RandomForestRegressor` and `GradientBoostingRegressor`.
- Binary targets (e.g., `shelf_life_exceeded`) use `RandomForestClassifier` and `GradientBoostingClassifier`.
- Evaluation: regression (R², MAE, RMSE), classification (accuracy, F1, ROC-AUC when available).

Recommended next improvements
----------------------------
- Add per-target hyperparameter tuning (GridSearchCV, RandomizedSearchCV, or Optuna).
- Replace single train/test split with k-fold cross-validation for robust metrics.
- Add experiment tracking (MLflow / W&B) to version models and metrics.
- Create a small inference script / API to load saved models and serve predictions.

Troubleshooting
---------------
- `Missing target columns`: make sure CSV contains the names in `target_columns` or run the estimator to synthesize them by importing `estimate_chocolate_targets`.
- `No numeric feature columns found`: ensure the CSV includes numeric formulation/process columns; otherwise add preprocessing to convert categorical columns.