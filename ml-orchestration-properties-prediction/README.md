# ML Orchestration — Dairy Properties Prediction

This repository contains experiments, models, and notebooks for predicting dairy product properties using classical machine learning models. It collects processed data, trained models (in ONNX format), metrics, and the main exploratory/processing notebook used for experiments.

## Repository Structure

- `Baladna_ml_orchestration_dairy_properties_prediction.ipynb` — primary analysis and experiment notebook.
- `data/` — datasets used by the notebook and experiments.
  - `raw/` — raw input files.
  - `processed/` — cleaned and feature-engineered datasets used for training/evaluation.
- `models/` — exported model artifacts and metrics (ONNX + JSON metrics files), e.g.:
  - `*_best_RandomForest.onnx`
  - `*_best_RandomForest_metrics.json`
- `src/` — (optional) supporting scripts and utilities used by the notebook or pipelines.

## Project Overview

The goal is to predict various dairy product properties (color channels, firmness, pH drift, etc.) using scikit-learn, LightGBM, KNN and other models. Trained models are exported to ONNX for portable inference and are bundled under `models/` alongside per-model metric JSON files.

## Getting Started

Prerequisites
- Python 3.8+ (recommended)
- `pip` for package installation

Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install required packages. If this repo does not include a `requirements.txt`, these are the minimum packages typically needed:

```powershell
pip install pandas numpy scikit-learn jupyterlab onnx onnxruntime matplotlib seaborn lightgbm
```

Start the notebook server:

```powershell
jupyter lab
# or
jupyter notebook
```

Open `Baladna_ml_orchestration_dairy_properties_prediction.ipynb` to explore preprocessing, model training, evaluation, and export steps.

## Data

Place raw dataset files under `data/raw/` and processed artifacts (if generated) under `data/processed/`. The notebook contains cells to read raw data and produce cleaned, feature-engineered CSVs used to train models.

## Models

Trained models and evaluation summaries live in `models/`. Each model is accompanied by a metrics JSON file. Models are exported to ONNX to enable fast, framework-agnostic inference. Example files:

- `color_a_best_RandomForest.onnx`
- `color_a_best_RandomForest_metrics.json`

## Quick Inference Example (Python + ONNX Runtime)

This example shows how to load an ONNX model and run a prediction using `onnxruntime`.

```python
import json
import numpy as np
import onnxruntime as rt

# Path to an ONNX model in `models/`
model_path = 'models/color_a_best_RandomForest.onnx'

sess = rt.InferenceSession(model_path)
input_name = sess.get_inputs()[0].name

# Example: create a single sample with appropriate features (replace with real features)
sample = np.array([[0.1, 1.2, 3.4, 5.6]], dtype=np.float32)

pred_onx = sess.run(None, {input_name: sample})
print('Prediction:', pred_onx)

# Load metrics for the model
with open('models/color_a_best_RandomForest_metrics.json', 'r') as f:
    metrics = json.load(f)
print('Metrics:', metrics)
```

Adjust the `sample` values to match the feature order used during model training.

## Reproducing Experiments

1. Ensure the dataset(s) are available in `data/raw/`.
2. Run the data-preparation cells in the notebook to generate `data/processed/` artifacts.
3. Run the model training and evaluation cells (or execute scripts in `src/` if present).
4. Export the best model to ONNX (the notebook shows example export steps).

## Contributing

Contributions are welcome. Recommended workflow:

1. Fork the repo and create a feature branch.
2. Add or update notebooks/scripts under `src/` and keep `Baladna_ml_orchestration_dairy_properties_prediction.ipynb` focused and documented.
3. Add tests if you add functionality or heavy logic.
4. Open a pull request describing your changes.

## License

Check for a `LICENSE` file in the repository root. If none is present, contact the repository owner before reusing code or models.

## Contact / Maintainers

Repository owner: `Badisj` (see repository metadata). For questions about the dataset, experiments, or models, open an issue or contact the owner/maintainers directly.