"""Train Random Forest and Gradient Boosting models for chocolate targets.

Usage:
    python -m src.train.train_targets --input data/complete_100_formulations_with_targets.csv \
        --output-dir outputs/models --test-size 0.2 --random-state 42

The script will:
- Extract `target_columns` from `src/process/2. Chocolate_target_estimation.py` programmatically.
- Load the CSV, build numeric feature matrix (drops target columns).
- For each target, train a Random Forest and Gradient Boosting model.
- Use a classifier for binary target `shelf_life_exceeded`, otherwise regressors.
- Save models with `joblib` and a `metrics.json` summary in the output directory.
"""
from pathlib import Path
import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import ast
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score


def extract_target_columns_from_source(source_path: Path):
    """Parse the Python source and extract the literal `target_columns` list.
    Uses AST to safely locate the assignment anywhere in the file.
    """
    src = source_path.read_text(encoding='utf-8')
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'target_columns':
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        raise RuntimeError('Found target_columns assignment but failed to evaluate it')
    raise RuntimeError('target_columns not found in source')


def prepare_data(df: pd.DataFrame, target_columns):
    # Ensure targets exist
    missing = [t for t in target_columns if t not in df.columns]
    if missing:
        raise ValueError(f"Missing target columns in input data: {missing}")

    # Use only numeric features and drop target columns from features
    numeric = df.select_dtypes(include=[np.number]).copy()
    X = numeric.drop(columns=target_columns, errors='ignore')
    # If X is empty, raise
    if X.shape[1] == 0:
        raise ValueError('No numeric feature columns found after dropping targets')

    return X, df[target_columns]


def train_and_evaluate(X, y, target_name, out_dir: Path, random_state=42, test_size=0.2, n_jobs=-1):
    is_classification = y.dropna().drop_duplicates().isin([0,1]).all()
    # Treat 'shelf_life_exceeded' as classification even if values present as 0/1
    if target_name == 'shelf_life_exceeded':
        is_classification = True

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if is_classification else None
    )

    results = {}

    if is_classification:
        models = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=n_jobs),
            'grad_boost': GradientBoostingClassifier(n_estimators=200, random_state=random_state)
        }
    else:
        models = {
            'random_forest': RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=n_jobs),
            'grad_boost': GradientBoostingRegressor(n_estimators=200, random_state=random_state)
        }

    for name, model in models.items():
        print(f"Training {name} for target: {target_name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metric = {}
        if is_classification:
            # For binary classification, compute accuracy, f1, roc_auc where available
            metric['accuracy'] = float(accuracy_score(y_test, preds))
            metric['f1'] = float(f1_score(y_test, preds, zero_division=0))
            try:
                # use predict_proba if available
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_test)[:, 1]
                    metric['roc_auc'] = float(roc_auc_score(y_test, probs))
                else:
                    metric['roc_auc'] = None
            except Exception:
                metric['roc_auc'] = None
        else:
            metric['r2'] = float(r2_score(y_test, preds))
            metric['mae'] = float(mean_absolute_error(y_test, preds))
            metric['rmse'] = float(mean_squared_error(y_test, preds, squared=False))

        # Save model
        model_path = out_dir / f"models" / f"{target_name}__{name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        # Persist metrics and some metadata
        results[name] = {
            'model_path': str(model_path),
            'metrics': metric,
            'n_features': X.shape[1]
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '--input-csv', dest='input_csv', required=True, help='Input CSV with features and targets')
    parser.add_argument('--process-source', dest='process_source', default='src/process/2. Chocolate_target_estimation.py', help='Path to the process file containing target_columns')
    parser.add_argument('--output-dir', dest='output_dir', default='outputs/models', help='Directory to save models and metrics')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_path}')

    process_path = Path(args.process_source)
    if not process_path.exists():
        raise FileNotFoundError(f'Process source not found: {process_path}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Extracting target columns from process file...')
    target_columns = extract_target_columns_from_source(process_path)
    print(f'Found {len(target_columns)} targets: {target_columns}')

    print('Loading input CSV...')
    df = pd.read_csv(input_path)

    print('Preparing data (numeric features only)...')
    X, Y = prepare_data(df, target_columns)

    all_results = {}
    for target in target_columns:
        print(f'\n=== Processing target: {target} ===')
        y = Y[target]
        try:
            res = train_and_evaluate(X, y, target, out_dir, random_state=args.random_state, test_size=args.test_size, n_jobs=args.n_jobs)
            all_results[target] = res
        except Exception as e:
            warnings.warn(f'Failed to train target {target}: {e}')

    # Save metrics summary
    metrics_path = out_dir / 'metrics_summary.json'
    metrics_path.write_text(json.dumps(all_results, indent=2))
    print(f'All done. Metrics written to: {metrics_path}')


if __name__ == '__main__':
    main()
