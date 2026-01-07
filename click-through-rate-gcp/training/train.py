import os
import argparse
import logging
import time
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_uri", required=True, help="GCS path to Parquet files")
    parser.add_argument("--label_col", default="label", help="Name of target column")
    args = parser.parse_args()

    logger.info("Starting training script...")
    start_total = time.time()

    # Start local Dask cluster (single-node multi-core)
    logger.info("Initializing Dask cluster...")
    client = Client(
        n_workers=8,
        threads_per_worker=1,
        memory_limit="6GB",
        processes=True,
    )
    logger.info(f"Dask cluster info: {client}")
    logger.info("Dask dashboard: %s", client.dashboard_link)

    # Load Parquet dataset from GCS
    logger.info(f"Reading Parquet from {args.data_uri}...")
    start = time.time()
    df = dd.read_parquet(args.data_uri, engine="pyarrow")
    logger.info(f"Parquet files loaded. Columns: {df.columns.tolist()}")
    logger.info(f"Row count (approx): {df.shape[0].compute()}")
    logger.info(f"Data load time: {time.time() - start:.2f} seconds")

    # Split features and target
    start = time.time()
    X = df.drop(columns=[args.label_col])
    y = df[args.label_col]
    logger.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Data preprocessing time: {time.time() - start:.2f} seconds")

    # Train/test split
    logger.info("Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Train a simple model
    logger.info("Training LogisticRegression model...")
    start = time.time()
    model = LogisticRegression(max_iter=500, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info(f"Training completed in {time.time() - start:.2f} seconds")

    # Evaluate
    acc = model.score(X_test, y_test)
    logger.info(f"Validation accuracy: {acc:.4f}")

    # Save model to the folder Vertex AI expects
    model_dir = os.environ.get("AIP_MODEL_DIR", "/tmp/model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info(f"Total training script time: {time.time() - start_total:.2f} seconds")
    logger.info("Training script finished successfully.")

if __name__ == "__main__":
    main()
