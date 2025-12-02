# Generative AI Orchestration (CPG)

Comprehensive repository for advanced formulation and orchestration of machine learning models for Continuous Processing and Generation (CPG). This project contains training and inference code, Jupyter notebooks for data processing and validation, and demo assets.

**Status:** Draft / Working repo — update manifests and dependencies for your environment.

**Table of Contents**
- **Overview**: High-level purpose and goals
- **Repository Structure**: Where code and notebooks live
- **Quick Start**: Create environment, install deps, run examples (PowerShell)
- **Training**: How to run training scripts and manifests
- **Inference**: Running inference scripts and manifests
- **Notebooks**: Data processing and model validation notebooks
- **Development**: How to contribute and run tests
- **Next Steps**: Suggested improvements and extensions
- **License & Contact**

**Overview**
- **Purpose**: Provide an organized codebase for building, training, and running ML models for advanced formulation problems including sensory, thermomechanical, and viscosity modeling.
- **Goals**: Reproducible training/inference pipelines, clear manifests for experiments, and quick interactive analysis using notebooks.

**Repository Structure**
Key folders and important files:

- `Demo/` : Presentation and demo project files (TS project files and recordings).
- `notebooks/`
  - `Data_Processing.ipynb` — data cleaning and preprocessing examples.
  - `Model_Validation.ipynb` — model evaluation and validation workflows.
- `src/`
  - `train/` — training scripts and manifests for each domain:
    - `Sensory/` — `train_sensory.py`, `Manifest_train_sensory.json`
    - `Thermomechanical/` — `train_thermomechanical.py`, `Manifest_train_thermomechanical.json`
    - `Viscosity/` — `train_viscosity.py`, `Manifest_train_viscosity.json`
  - `inference/` — inference scripts and manifests:
    - `Sensory/` — `inference_sensory.py`, `Manifest_inference_sensory.json`
    - `Thermomechanical/` — `inference_thermomechanical.py`, `Manifest_inference_thermomechanical.json`
    - `Viscosity/` — `inference_viscosity.py`, `Manifest_inference_viscosity.json`
  - `processing/` — data processing utilities: `process.py`, `Manifest_process.json`.

Notes:
- Each domain folder typically contains a manifest JSON describing dataset paths, hyperparameters, or deployment settings. Adjust paths in manifests to match your local data layout.

**Quick Start (PowerShell)**
1. Open a PowerShell terminal in the project root (for example, `c:\Users\mji11\Desktop\Machine Learning Orchestration\ai-ml-orchestration-cpg\ml-advanced-formulation`).
2. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. (Optional) Create a `requirements.txt` if not present, then install dependencies. Example:

```powershell
# If you have a requirements.txt
pip install -r requirements.txt

# OR install common ML packages (adjust versions as needed)
pip install numpy pandas scikit-learn torch torchvision jupyterlab matplotlib seaborn
```

4. Run a training script example (edit manifest paths where required):

```powershell
python src\train\Sensory\train_sensory.py --manifest src\train\Sensory\Manifest_train_sensory.json
```

5. Run an inference script example:

```powershell
python src\inference\Viscosity\inference_viscosity.py --manifest src\inference\Viscosity\Manifest_inference_viscosity.json
```

6. Launch the notebooks:

```powershell
jupyter lab
# or
jupyter notebook
```

**Training**
- Each `train_*.py` accepts a manifest describing dataset locations and hyperparameters. Open the corresponding `Manifest_train_*.json` and update dataset paths and output directories.
- Typical workflow:
  - Prepare data with `src/processing/process.py` or the `notebooks/Data_Processing.ipynb`.
  - Edit the manifest to point to preprocessed data.
  - Run the training script and monitor outputs (saved model weights, logs, metrics).

**Inference**
- Inference scripts follow a similar manifest-driven approach. Update `Manifest_inference_*.json` with the path to saved weights and the input data.
- Example command (PowerShell):

```powershell
python src\inference\Thermomechanical\inference_thermomechanical.py --manifest src\inference\Thermomechanical\Manifest_inference_thermomechanical.json
```

**Notebooks**
- `Data_Processing.ipynb`: sample ETL and feature preparation steps. Use it to adapt preprocessing to your datasets.
- `Model_Validation.ipynb`: run post-training validation, visualizations, and metric analysis.

**Development & Contribution**
- Contribution guidelines (suggested):
  - Fork the repository, create a feature branch, and open a pull request with a clear description.
  - Add tests for any new processing/training/inference code.
  - Keep manifests and sample configs small and documented.
- Recommended additions:
  - Add a `requirements.txt` or `pyproject.toml` for reproducible environments.
  - Add CI for linting and running unit tests (GitHub Actions).

**Next Steps (Suggested)**
- Add sample (small) datasets and example manifests for quick verification.
- Provide Dockerfile(s) or `devcontainer.json` for consistent development environments.
- Implement a `scripts/` wrapper for common commands: `train-all`, `run-inference`, `eval`.

**License & Contact**
- License: Add a `LICENSE` file to set the repo license (e.g., MIT, Apache-2.0). Currently unspecified.
- Contact: For questions or collaboration, open an issue or contact the repository owner.
