# Beauty & Personal Care - Generative Formulation

This project contains machine learning models and inference scripts for beauty and personal care product formulation, including microbiology, physiochemical properties, and stability analysis.

## Project Structure

```
.
├── notebooks/
│   ├── Microbiology_Model_Training.ipynb
│   └── Physiochemical_Properties_Model_Training.ipynb
└── src/
    ├── Inference_Microbiology.py
    ├── Inference_Physiochemical.py
    └── Inference_Stability.py
```

## Components

### Notebooks
- **Microbiology_Model_Training.ipynb** - Training pipeline for microbiology prediction models
- **Physiochemical_Properties_Model_Training.ipynb** - Training pipeline for physiochemical properties prediction models

### Inference Scripts
- **Inference_Microbiology.py** - Inference module for microbiology predictions
- **Inference_Physiochemical.py** - Inference module for physiochemical property predictions
- **Inference_Stability.py** - Inference module for stability analysis predictions

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required dependencies (see requirements.txt or environment setup)

### Usage

1. **Training Models**
   - Open and run the Jupyter notebooks in `notebooks/` to train the ML models
   - Notebooks contain data preprocessing, feature engineering, model training, and evaluation

2. **Running Inference**
   - Use the inference scripts in `src/` to make predictions:
     ```python
     python src/Inference_Microbiology.py
     python src/Inference_Physiochemical.py
     python src/Inference_Stability.py
     ```

## Features

- Microbiology prediction for product safety and shelf life
- Physiochemical property prediction for formulation optimization
- Stability analysis for product performance
- Modular inference scripts for easy integration

## Documentation

For detailed information about model architecture, training procedures, and API references, refer to the individual notebook and script files.

## License

Part of the Generative AI Orchestration CPG project by Badisj.
