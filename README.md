# Biological Data Model Training Framework

This framework provides a comprehensive pipeline for training machine learning models on biological data, featuring ensemble feature selection and TabPFN models for prediction. It's designed to handle various types of biomarker data, with particular focus on omics data analysis and target prediction.

## Project Structure

```
├── data_loader.py       # Data loading and preprocessing utilities
├── feature_selection.py # Ensemble feature selection implementation
├── model_training.py    # Model training and evaluation logic
├── main.py              # Main execution script for training
├── prediction.py        # Module for making predictions with trained models
├── requirements.txt     # Package dependencies
└── README.md            # Project documentation
```

## Features

- **Ensemble Feature Selection**: Combines multiple methods (Mutual Information, XGBoost, CatBoost, Random Forest) to select the most informative features
- **Multi-task Learning**: Supports both regression and classification tasks
- **GPU Acceleration**: Optional GPU support for XGBoost and CatBoost
- **Covariate Integration**: Optional inclusion of covariates in models
- **Person-level Validation**: Ensures train/test splits respect person identity (no data leakage)
- **TabPFN Models**: Utilizes Transformer-based TabPFN models for accurate tabular data prediction
- **Target Grouping**: Organizes targets into biological categories (longitudinal, pathology, omics, etc.)
- **Prediction Module**: Dedicated module for making predictions with trained models
- **Feature Scaling Control**: Option to enable or disable StandardScaler on input features

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Data Structure

The framework expects the following data structure in the data directory:

- `predictors.txt.gz`: Predictor variables (e.g., proteins or other omics data)
- `predictor_ids.txt`: IDs for predictors (optional)
- `targets.txt.gz`: Target variables to predict
- `target_annotation.txt`: Metadata for targets including type and grouping
- `covs.txt.gz`: Covariate data (optional)
- `cov_annotation.txt`: Metadata for covariates (optional)

All files should use tab-separated values format with consistent sample IDs.

## Target and Feature Categories

The framework organizes targets and features into the following categories:

### Target Categories
- **longitudinal**: Time-series measurements collected over multiple visits
- **pathology**: Disease-related measures and clinical outcomes
- **omics**: Molecular data like genomics, proteomics, etc.
- **demographic**: Age, sex, ethnicity, etc.
- **genetic**: Genetic markers and variants
- **slope**: Rate of change measures over time

### Feature Categories
- **predictor**: Features used to predict targets, typically 'omics' data
- **covariate**: Additional variables that may influence the target, typically 'demographic' data

## Usage

### Training Models

Basic usage:
```bash
python main.py --method CatBoost
```

With covariates:
```bash
python main.py --include_covariates --method CatBoost
```

With feature scaling (applies StandardScaler to input features):
```bash
python main.py --scale_features --method CatBoost
```

Specify targets:
```bash
python main.py --targets cts_mmse30 msex age_at_visit
```

Process all targets in a specific group:
```bash
python main.py --target_group omics
```

Process all targets from multiple groups:
```bash
python main.py --target_group all
```

With custom directories:
```bash
python main.py --data_dir ../data --output_dir ./output --method CatBoost
```

### Making Predictions

Basic prediction:
```bash
python prediction.py --models_dir ./models --output_dir ./predictions
```

Force using covariates for all models:
```bash
python prediction.py --force_covariates
```

Force feature scaling for all models:
```bash
python prediction.py --force_scaling
```

Force skipping feature scaling for all models:
```bash
python prediction.py --force_no_scaling
```

Predict only specific targets:
```bash
python prediction.py --target_ids target1_id target2_id
```

### Command-line Arguments

#### Training (main.py)
- `--include_covariates`: Include covariates in models
- `--scale_features`: Apply StandardScaler to input features
- `--method`: Feature selection method (default: 'CatBoost', options: 'MI', 'XGB', 'CatBoost', 'RF')
- `--targets`: Specific targets to process
- `--target_group`: Target group to process (choices: 'longitudinal', 'pathology', 'omics', 'demographic', 'genetic', 'slope', 'all')
- `--data_dir`: Directory containing data files (default: '../data')
- `--output_dir`: Directory for output files (default: '.')
- `--verbose`: Verbosity level (default: 1)

#### Prediction (prediction.py)
- `--data_dir`: Directory containing data files (default: '../data')
- `--models_dir`: Directory containing model files (default: './models')
- `--output_dir`: Directory to save predictions (default: './predictions')
- `--force_covariates`: Force using covariates for all models
- `--force_no_covariates`: Force excluding covariates for all models
- `--force_scaling`: Force using feature scaling for all models
- `--force_no_scaling`: Force skipping feature scaling for all models
- `--no_skip_existing`: Do not skip targets with existing predictions
- `--target_ids`: Specific target IDs to process

## Output Files

The framework generates the following outputs:

- `models/`: Directory containing trained models (pickle files)
- `results/`: Directory containing individual target results (JSON)
- `regression_results.csv`: Summary of regression model performances
- `classification_results.csv`: Summary of classification model performances
- `all_results_summary.csv`: Combined summary of all results
- `predictions/`: Directory containing predictions (when using prediction.py)

## Feature Selection Details

The feature selection module implements an ensemble approach that combines multiple methods:

- **Mutual Information**: Information-theoretic approach for measuring feature relevance
- **XGBoost**: Feature importance from gradient boosted trees
- **CatBoost**: Feature importance from gradient boosted trees with categorical feature support
- **Random Forest**: Feature importance from random forest ensembles

Features are ranked based on their aggregate importance across all selected methods, and the top k features are selected.

## Feature Scaling

The framework provides flexible control over feature scaling:

- During training, use `--scale_features` to apply StandardScaler to input features
- Models save their scaling preference in the model file
- During prediction, the system automatically applies scaling based on model requirements
- You can override scaling behavior with `--force_scaling` or `--force_no_scaling`

This allows you to experiment with different scaling approaches and optimize model performance.

## Model Training Details

The framework uses TabPFN (Tabular Prior-Data Fitted Networks) models which leverage transformer architectures pre-trained on thousands of synthetic tabular datasets. These models offer:

- Strong performance on both regression and classification tasks
- Ability to handle categorical features
- Fast training and inference
- Built-in handling of feature interactions

## Performance Metrics

- **Regression**: R², MSE, MAE, Pearson correlation
- **Classification**: Accuracy, Precision, Recall, F1 score, Confusion matrix

## GPU Support

The framework supports GPU acceleration for feature selection methods that can utilize it (XGBoost, CatBoost) and for TabPFN models when PyTorch CUDA is available.

## Requirements

Key dependencies include:
- numpy, pandas
- scikit-learn
- xgboost, catboost
- tabpfn
- torch

See requirements.txt for complete dependencies.

## License

[Specify license here]