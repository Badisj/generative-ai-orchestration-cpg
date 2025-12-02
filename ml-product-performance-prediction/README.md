# ML Product Performance Prediction — Training Targets

**Overview**
- **Description**: This repository contains tools to estimate chocolate product target variables (sensory, physical, stability) and to train machine-learning models (Random Forest and Gradient Boosting) to predict those targets from formulation and process features.
- **Primary script**: `src/train/train_targets.py` — trains models for each target defined in `src/process/2. Chocolate_target_estimation.py` and saves models + metrics.

**Requirements**
- **Python**: 3.8+ recommended.
- **Packages**: Install from `requirements.txt`.

**Installation**
- **Create venv (Windows PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Usage**
- **Basic training command (PowerShell)**:
```powershell
python -m src.train.train_targets --input data/Chocolate_bar_dataset_with_features.csv --output-dir outputs/models --test-size 0.2 --random-state 42
```
- **Arguments**:
  - `--input`: Path to input CSV containing both features and the target columns (required).
  - `--process-source`: Path to `2. Chocolate_target_estimation.py` (default: `src/process/2. Chocolate_target_estimation.py`). The script reads `target_columns` from this file.
  - `--output-dir`: Directory where models and `metrics_summary.json` will be saved (default: `outputs/models`).
  - `--test-size`: Fraction for the test split (default: `0.2`).
  - `--random-state`: Random seed for reproducibility (default: `42`).

**Input Data Expectations**
- **Format**: CSV file with one row per formulation.
- **Features**: Numeric columns (formulation percentages, process parameters, indices). The training script selects numeric columns automatically and drops any columns listed in `target_columns`.
- **Targets**: The `target_columns` list is defined in `src/process/2. Chocolate_target_estimation.py`. Example targets include `hardness_newtons`, `viscosity_pas`, `overall_preference`, and `shelf_life_exceeded` (binary).

**Outputs**
- **Model files**: Saved as Joblib files at `<output-dir>/models/<target_name>__<model_name>.joblib`.
- **Metrics**: Summary JSON written to `<output-dir>/metrics_summary.json` containing per-target, per-model metrics and saved model paths.

**Modeling Details**
- **Regression targets**: Models use `RandomForestRegressor` and `GradientBoostingRegressor` with 200 estimators by default.
- **Classification targets**: Binary targets (e.g., `shelf_life_exceeded`) use `RandomForestClassifier` and `GradientBoostingClassifier`.
- **Evaluation**: Regression metrics include R², MAE, RMSE. Classification metrics include accuracy, F1 and ROC-AUC (where available).

**Extending & Improvements**
- **Hyperparameter tuning**: Integrate `GridSearchCV` or `RandomizedSearchCV` for per-target tuning.
- **Cross-validation**: Replace single train/test split with k-fold CV for more stable metrics.
- **Feature engineering**: Add domain-specific features, scaling, or PCA in a preprocessing pipeline.
- **Experiment tracking**: Add MLflow or Weights & Biases to track runs, parameters and artifacts.

**Developer Notes**
- The script extracts `target_columns` by parsing the AST of `src/process/2. Chocolate_target_estimation.py`. Ensure that file contains a literal list assignment named `target_columns` (the default implementation does).
- If your dataset uses categorical features, convert or encode them before running the training script (the script currently uses numeric columns only).

**Troubleshooting**
- `No numeric feature columns found`: Ensure your CSV includes numeric feature columns and that target columns are present.
- `Missing target columns`: Confirm the input CSV contains all names listed under `target_columns` in the process file.

**Next steps I can help with**
- Add `requirements-dev.txt` and CI tests.
- Implement hyperparameter search and cross-validation.
- Add a Jupyter notebook demo showing loading models and producing predictions.
