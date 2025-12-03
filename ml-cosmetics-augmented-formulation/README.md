# ML Augmented Formulation

Compact README for the `ml-augmented-formulation` utilities used in the generative AI orchestration project.

**Project purpose:**
- **Summary:** Implements training and inference scripts for two modelling tasks — microbiology and stability — used in ML-augmented formulation research and orchestration experiments.
- **Intended users:** Researchers and engineers who want to run training and inference locally or integrate these scripts into an orchestration pipeline.

**Repository layout:**
- `data/`: Place datasets and derived artifacts here.
- `src/`: Source scripts for training and inference.
  - `train_microbiology.py` — training entry for microbiology models.
  - `train_stability.py` — training entry for stability models.
  - `inference_microbiology.py` — inference/serving helper for microbiology models.
  - `inference_stability.py` — inference/serving helper for stability models.

Requirements
- **Python:** 3.8+ recommended.
- **Dependencies:** If a `requirements.txt` is present at the project root, install it (instructions below). If not, ask to add one or inspect `src/` for imports to build one.

Quick setup (PowerShell)

```powershell
# create a virtual environment
python -m venv .venv
# activate the venv (PowerShell)
.\.venv\Scripts\Activate.ps1
# install dependencies if a requirements file exists
if (Test-Path requirements.txt) { pip install -r requirements.txt }
```

Running training
- **Microbiology training:**
  - `python src\train_microbiology.py --help` — shows available CLI options (dataset path, model output dir, hyperparams).
  - Example:
    ```powershell
    python src\train_microbiology.py --data data/microbiology.csv --out models/microbiology
    ```
- **Stability training:**
  - `python src\train_stability.py --help`
  - Example:
    ```powershell
    python src\train_stability.py --data data/stability.csv --out models/stability
    ```

Running inference
- **Microbiology inference (local):**
  - `python src\inference_microbiology.py --help`
  - Example:
    ```powershell
    python src\inference_microbiology.py --model models/microbiology --input data/experiment_input.csv --output predictions/microbiology_preds.csv
    ```
- **Stability inference (local):**
  - `python src\inference_stability.py --model models/stability --input data/experiment_input.csv --output predictions/stability_preds.csv`

Data expectations
- Put raw or preprocessed datasets under `data/`. The training scripts generally expect a tabular CSV-style file; check each script's `--help` output to confirm required columns and preprocessing steps.

Development notes
- There are no automated tests or CI configured by default in this repository. Consider adding a `requirements.txt` and a minimal `pytest` suite for reproducibility.
- Add a `LICENSE` file if you want to publish or share this project publicly.