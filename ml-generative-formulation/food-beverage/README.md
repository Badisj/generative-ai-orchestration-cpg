# Dairy Generative Formulation

A machine learning orchestration project for predicting sensory attributes and nutritional properties in dairy product formulations using AWS SageMaker.

## Project Overview

This project implements an end-to-end ML pipeline for yogurt product formulation. It leverages sensory evaluation data, nutritional information, and cost metrics to train predictive models that can generate optimized product formulations.

## Directory Structure

```
.
├── data/
│   ├── raw/
│   │   └── yogurt_data_sensory_nutrition_cost_raw.csv          # Raw input data
│   └── processed/
│       ├── yogurt_data_sensory_nutrition_cost_train.csv         # Training dataset
│       ├── yogurt_data_sensory_nutrition_cost_validation.csv    # Validation dataset
│       ├── yogurt_data_sensory_nutrition_cost_test.csv          # Test dataset
│       └── encoder/                                              # Categorical encoders (pickled)
├── notebooks/
│   └── Dairy Generative Formulation.ipynb                       # Exploratory analysis & experimentation
├── src/
│   ├── processing.py                                             # Data preprocessing & feature engineering
│   ├── training.py                                               # Model training pipeline
│   ├── inference.py                                              # Model inference & prediction
│   └── evaluate_metrics.py                                       # Evaluation metrics utilities
└── README.md
```

## Components

### 1. Data Processing (`src/processing.py`)

Handles data preparation and feature engineering:

- **Input**: Raw CSV file with recipe data, sensory attributes, and nutritional information
- **Processing Steps**:
  - Categorical encoding using `LabelEncoder` for object columns
  - Train/validation/test split (configurable ratios)
  - Feature store integration with AWS SageMaker Feature Store
  - Encoder persistence for inference pipeline
- **Output**: Train, validation, and test CSV files with encoded features

**Key Functions**:
- `process()`: Main data processing pipeline
- `create_feature_group()`: Create AWS Feature Store feature group
- `wait_for_feature_group_creation()`: Monitor feature group status
- `cast_object_to_string()`: Type casting for feature store compatibility

### 2. Model Training (`src/training.py`)

Trains Random Forest Regressors for sensory attribute prediction:

- **Algorithm**: Random Forest Regressor (scikit-learn)
- **Target Sensory Attributes**:
  - Flavor intensity
  - Sweetness
  - Fruit intensity
  - Chalkiness
  - Color intensity
  - Thickness
  - Coating
  - Global Appreciation

**Key Features**:
- Configurable hyperparameters (n_estimators, max_depth, criterion)
- Multi-target model training
- Model validation with comprehensive metrics (MSE, RMSE, MAE, R²)
- Model serialization using joblib

**Hyperparameter Options**:
- `--n_estimators`: Number of trees (default: 100)
- `--max_depth`: Maximum tree depth (default: 3)
- `--criterion`: Splitting criterion (default: 'gini')
- `--random_state`: Random seed (default: 2024)
- `--sensory_output`: Target sensory attribute to predict

### 3. Inference (`src/inference.py`)

Deployment-ready inference module for SageMaker:

- **Functions**:
  - `model_fn()`: Loads compressed model from tar.gz archive
  - `predict_fn()`: Generates predictions on new data
- **Output**: Predictions as pandas Series

### 4. Evaluation Metrics (`src/evaluate_metrics.py`)

Utilities for model evaluation and performance tracking.

## Usage

### Prerequisites

```bash
pip install pandas scikit-learn joblib boto3 sagemaker
```

### Data Processing

```bash
python src/processing.py \
  --input-data /path/to/raw/data \
  --output-data /path/to/output \
  --validation-split-percentage 0.10 \
  --test-split-percentage 0.20 \
  --feature-store-offline-prefix "dairy-formulation" \
  --feature-group-name "dairy-sensory-features"
```

### Model Training

```bash
python src/training.py \
  --n_estimators 100 \
  --max_depth 3 \
  --criterion gini \
  --random_state 2024 \
  --sensory_output sweetness
```

### AWS SageMaker Integration

The pipeline is designed to work with AWS SageMaker:

- **Processing Jobs**: Use `processing.py` for data preprocessing
- **Training Jobs**: Use `training.py` for model training with SageMaker Training
- **Deployment**: Use `inference.py` with SageMaker Endpoints

Required environment variables for SageMaker:
- `SM_HOSTS`: List of hosts running the training
- `SM_CURRENT_HOST`: Current host name
- `SM_MODEL_DIR`: Model output directory
- `SM_CHANNEL_TRAIN`: Training data channel
- `SM_CHANNEL_VALIDATION`: Validation data channel
- `SM_OUTPUT_DIR`: Output directory
- `SM_NUM_GPUS`: Number of GPUs available

## Sensory Attributes

The model predicts the following sensory attributes based on formulation inputs:

1. **Flavor Intensity** - Strength of flavor profile
2. **Sweetness** - Perceived sweetness level
3. **Fruit Intensity** - Intensity of fruit flavors
4. **Chalkiness** - Chalky texture perception
5. **Color Intensity** - Visual color strength
6. **Thickness** - Texture/consistency
7. **Coating** - Mouthfeel coating sensation
8. **Global Appreciation** - Overall product score

## Model Performance Metrics

The training pipeline evaluates models using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## Data Pipeline

```
Raw Data
    ↓
[Categorical Encoding]
    ↓
[Train/Val/Test Split]
    ↓
[Feature Store Ingestion]
    ↓
[Model Training]
    ↓
[Model Evaluation]
    ↓
[Model Deployment]
```

## Notebook

`notebooks/Dairy Generative Formulation.ipynb` - Jupyter notebook for:
- Exploratory data analysis
- Feature exploration
- Model experimentation
- Results visualization

## Key Workflows

### Training a New Model

1. Prepare raw data in `data/raw/`
2. Run `processing.py` to preprocess and split data
3. Run `training.py` with desired hyperparameters
4. Review metrics and model performance
5. Save model artifacts

### Making Predictions

1. Load trained model using `model_fn()` from `inference.py`
2. Prepare input features with same encoding as training data
3. Call `predict_fn()` to generate predictions
4. Post-process results as needed

## Configuration Files

The following files should be present in the inference deployment:
- `inference.py` - Inference code
- `requirements.txt` - Python dependencies
- `config.json` - Configuration parameters

## Notes

- Models are persisted as joblib files for reproducibility
- Categorical encoders are saved separately for use during inference
- The pipeline is designed for AWS SageMaker orchestration
- Feature Store integration enables real-time feature serving

## Future Enhancements

- Model hyperparameter tuning and grid search
- Additional regression algorithms (XGBoost, Neural Networks)
- Real-time prediction endpoint deployment
- Model versioning and A/B testing
- Cross-validation and ensemble methods
