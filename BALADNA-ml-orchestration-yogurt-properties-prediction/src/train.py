#!/usr/bin/env python3
"""
train_ml_yogurt_prod.py

Production-grade high-level ML training for yogurt dataset.
- Trains candidate regressors per target (RandomForest, HistGB, KNN, SVR).
- Selects the best model per target using CV RMSE.
- Saves best models in ONNX format.
- Saves metrics and best hyperparameters in JSON.
- Uses robust logging.

Requires:
    pip install scikit-learn scikit-optimize skl2onnx onnxmltools onnxruntime
"""

import argparse
import os
import json
import logging
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from skopt import BayesSearchCV
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ------------------------------
# Suppress warnings for cleaner logs
# ------------------------------
warnings.filterwarnings("ignore")


# ------------------------------
# Logging configuration
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("YogurtML")


# ------------------------------
# Default targets in dataset
# ------------------------------
DEFAULT_TARGETS = [
    "pH_evolution",
    "viscosity",
    "fat_globule_size_um",
    "color_L_star",
    "color_a_star",
    "color_b_star",
    "yeast_mold_growth_log_CFU",
    "overall_liking_score",
    "firmness_N",
    "syneresis_ml_per_100g",
    "ph_drift_14days",
]


# ------------------------------
# CLI argument parser
# ------------------------------
def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train ML models with CV + Bayesian search (ONNX export)")
    parser.add_argument("--input", "-i", type=str, required=True, help="CSV dataset path")
    parser.add_argument("--targets", "-t", nargs="+", default=DEFAULT_TARGETS, help="Target column names")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--out-dir", type=str, default="./models", help="Output directory for models + metrics")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs")
    parser.add_argument("--trials", type=int, default=30, help="BayesSearch iterations per model")
    return parser.parse_args()


# ------------------------------
# Load dataset and split features/targets
# ------------------------------
def load_data(path, targets):
    """Load dataset from CSV and split into X/y."""
    df = pd.read_csv(path)
    X = df.drop(columns=targets)
    y = df[targets]
    return X, y


# ------------------------------
# Save scikit-learn model to ONNX
# ------------------------------
def save_model_to_onnx(model, X, outpath):
    n_features = X.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open(outpath, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logger.info(f"ONNX model saved: {outpath}")
    except Exception as e:
        logger.error(f"ONNX export failed for {outpath.name}: {e}")


# ------------------------------
# Evaluate model using CV
# ------------------------------
def evaluate_model_cv(model, X, y, cv):
    """
    Compute regression metrics using cross-validation predictions.
    Returns dict with averaged RMSE, MAE, R² across all targets.
    """
    preds = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    rmse = root_mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}


# ------------------------------
# Define candidate models and hyperparameter search spaces
# ------------------------------
def get_model_spaces(random_state=42):
    return {
        "RandomForest": (
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            {"model__n_estimators": (100, 1000), "model__max_depth": (3, 30)}
        ),

        "GradientBoosting": (GradientBoostingRegressor(random_state=random_state),
        {
            "model__n_estimators": (100, 1000),
            "model__learning_rate": (1e-3, 0.5, "log-uniform"),
            "model__max_depth": (3, 15),
            "model__subsample": (0.5, 1.0, "uniform"),
            "model__min_samples_split": (2, 20),
        },
    ),

        "HistGB": (
            HistGradientBoostingRegressor(random_state=random_state),
            {"model__max_iter": (50, 500), "model__learning_rate": (1e-3, 0.5, "log-uniform")}
        ),

        "LightGBM": (
        LGBMRegressor(random_state=random_state),
        {
            "model__n_estimators": (200, 2000),
            "model__learning_rate": (1e-3, 0.5, "log-uniform"),
            "model__max_depth": (3, 20),
            "model__reg_alpha": (1e-4, 10.0, "log-uniform"),
            "model__reg_lambda": (1e-4, 10.0, "log-uniform"),
        },
    ),

        "KNN": (
            KNeighborsRegressor(),
            {"model__n_neighbors": (2, 50)}
        ),

        "SVR": (
            SVR(),
            {"model__C": (1e-2, 1e3, "log-uniform"), "model__epsilon": (1e-4, 1.0, "log-uniform")}
        ),
    }


# ------------------------------
# Main training loop (per-target)
# ------------------------------
def main():
    summary = []
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load dataset
    X, y_all = load_data(args.input, args.targets)
    logger.info(f"Data loaded: X={X.shape}, y={y_all.shape}")

    # Clean dataset
    X = X.dropna(axis=1, how="all")
    mask = ~y_all.isna().any(axis=1)
    X, y_all = X.loc[mask], y_all.loc[mask]
    logger.info(f"After cleaning: X={X.shape}, y={y_all.shape}")

    # CV splitter
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    # Candidate models + hyperparameter search spaces
    model_spaces = get_model_spaces(random_state=args.random_state)

    # Iterate over targets
    for target in args.targets:
        y = y_all[[target]]
        logger.info(f"========================== Training target: {target} ==========================")

        best_rmse = float("inf")
        best_model = None
        best_metrics = None
        best_name = None

        for name, (estimator, search_space) in model_spaces.items():
            logger.info(f"-------------------------- Model: {name} --------------------------")

            # Pipeline
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if name in ["KNN", "SVR"]:
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", estimator))
            pipe = Pipeline(steps)

            # Bayesian search
            opt = BayesSearchCV(
                pipe,
                search_spaces=search_space,
                n_iter=args.trials,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=args.n_jobs,
                random_state=args.random_state,
                verbose=0,
            )

            # Fit
            opt.fit(X, y.values.ravel())
            best_model = opt.best_estimator_

            # Compute metrics
            metrics = evaluate_model_cv(best_model, X, y, cv)
            metrics["best_params"] = opt.best_params_
            metrics["best_score_cv"] = -opt.best_score_
            logger.info(f"{name} CV RMSE: {metrics['RMSE']:.4f}")
            logger.info(f"{name} CV R2: {metrics['R2']:.4f}")

            # Keep the best
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                bestR2 = metrics["R2"]
                best_model = opt.best_estimator_
                best_metrics = metrics
                best_name = name

        # Save best model ONNX and metrics JSON
        onnx_path = Path(args.out_dir) / f"{target}_best_{best_name}.onnx"
        json_path = Path(args.out_dir) / f"{target}_best_{best_name}_metrics.json"
        save_model_to_onnx(best_model, X, onnx_path)
        with open(json_path, "w") as f:
            json.dump(best_metrics, f, indent=4)
        logger.info(f"Best model for {target}: {best_name}  RMSE={best_rmse:.4f}")
        logger.info(f"ONNX saved to {onnx_path}, metrics saved to {json_path}")

        # Append to summary
        summary.append({
            "target": target,
            "best_model": best_name,
            "R2": bestR2,
            "rmse": best_rmse,
            "onnx_path": str(onnx_path),
            "metrics_json": str(json_path)
        })

    # Print final summary
    logger.info("\n=== Final Summary ===")
    for entry in summary:
        logger.info(
            f"Target: {entry['target']:<20} "
            f"Best Model: {entry['best_model']:<12} "
            f"R2: {entry['R2']:.4f} "
            f"RMSE: {entry['rmse']:.4f} "
            f"ONNX: {entry['onnx_path']} "
            f"Metrics: {entry['metrics_json']}"
        )

            
    

# ------------------------------
# Main training loop (multi-target)
# ------------------------------

# def main():
#     args = parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     # Load dataset
#     X, y = load_data(args.input, args.targets)
#     logger.info(f"Data loaded: X={X.shape}, y={y.shape}")

#     # Clean dataset: drop NaN-only features + rows with NaN targets
#     X = X.dropna(axis=1, how="all")
#     mask = ~y.isna().any(axis=1)
#     X, y = X.loc[mask], y.loc[mask]
#     logger.info(f"After cleaning: X={X.shape}, y={y.shape}")

#     # Define CV splitter
#     cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

#     # Candidate models + hyperparameter search spaces
#     model_spaces = {
#         "RandomForest": (
#             RandomForestRegressor(random_state=args.random_state, n_jobs=1),
#             {
#                 "model__estimator__n_estimators": (100, 1000),
#                 "model__estimator__max_depth": (3, 30),
#             },
#         ),
#         "HistGB": (
#             HistGradientBoostingRegressor(random_state=args.random_state),
#             {
#                 "model__estimator__max_iter": (100, 1000),
#                 "model__estimator__learning_rate": (1e-3, 0.5, "log-uniform"),
#             },
#         ),
#         "KNN": (
#             KNeighborsRegressor(),
#             {
#                 "model__estimator__n_neighbors": (2, 50),
#             },
#         ),
#         "SVR": (
#             SVR(),
#             {
#                 "model__estimator__C": (1e-2, 1e3, "log-uniform"),
#                 "model__estimator__epsilon": (1e-4, 1.0, "log-uniform"),
#             },
#         ),
#     }

#     # Store results summary
#     results = []

#     for name, (estimator, search_space) in model_spaces.items():
#         logger.info(f"=== Training {name} ===")

#         # Wrap estimator in MultiOutputRegressor for multi-target support
#         multi = MultiOutputRegressor(estimator)

#         # Build preprocessing + model pipeline
#         steps = [("imputer", SimpleImputer(strategy="median"))]
#         if name in ["KNN", "SVR"]:  # these models need scaling
#             steps.append(("scaler", StandardScaler()))
#         steps.append(("model", multi))
#         pipe = Pipeline(steps)

#         # Hyperparameter optimization with Bayesian search
#         opt = BayesSearchCV(
#             pipe,
#             search_spaces=search_space,
#             n_iter=args.trials,
#             cv=cv,
#             scoring="neg_root_mean_squared_error",
#             n_jobs=args.n_jobs,
#             random_state=args.random_state,
#             verbose=0,
#         )

#         # Train and optimize
#         opt.fit(X, y)
#         best_model = opt.best_estimator_
#         logger.info(f"{name} best CV RMSE: {-opt.best_score_:.4f}")
#         logger.debug(f"{name} best params: {opt.best_params_}")

#         # Save ONNX
#         onnx_path = Path(args.out_dir) / f"best_{name}.onnx"
#         save_model_to_onnx(best_model, X, onnx_path)

#         # Compute and save metrics JSON
#         metrics = evaluate_model_cv(best_model, X, y, cv)
#         metrics["best_params"] = opt.best_params_
#         metrics["best_score_cv"] = -opt.best_score_

#         json_path = Path(args.out_dir) / f"best_{name}_metrics.json"
#         with open(json_path, "w") as f:
#             json.dump(metrics, f, indent=4)
#         logger.info(f"Metrics JSON saved: {json_path}")

#         results.append((name, metrics["RMSE"], onnx_path, json_path))

#     # Print summary
#     logger.info("=== Final Results ===")
#     for name, rmse, path, jsonpath in sorted(results, key=lambda x: x[1]):
#         logger.info(f"{name:<12} RMSE={rmse:.4f}  model={path}  metrics={jsonpath}")


if __name__ == "__main__":
    main()
