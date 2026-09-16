# Biological Data Model Training Framework

**Built with PriorLabs-TabPFN.** This pipeline code is released under the MIT License; the TabPFN models and weights it uses are covered by the [Prior Labs License](LICENSE-TabPFN). See [License](#license).

This framework provides a comprehensive pipeline for training machine learning models on biological data, featuring ensemble feature selection and TabPFN models for prediction. It's designed to handle various types of biomarker data, with particular focus on omics data analysis and target prediction.


## Method description
# TabPFN Pipeline Method Description

Both regression and classification models were implemented using TabPFN (Tabular Prior-Fitted Networks), a transformer-based approach that leverages pre-trained neural networks for tabular data prediction. The pipeline incorporated feature selection, data preprocessing, and model training with the following methodology:

**Feature Selection and Preprocessing**: When the number of input features exceeded 500, CatBoost-based feature selection was applied to reduce dimensionality to ≤500 features before TabPFN modeling. Selected features were combined with covariates when specified, and both predictors and targets were standardized to ensure consistent scaling across variables.

**Model Configuration**: For regression tasks, TabPFN regressors were configured with GPU acceleration and 8 ensemble estimators. For classification tasks, TabPFN classifiers were similarly configured with GPU acceleration and 8 ensemble estimators. Categorical feature indices were explicitly specified to handle mixed data types appropriately. Both models utilized pre-trained weights optimized for their respective tabular tasks.

**Training and Prediction**: For regression, models were trained on scaled target variables and generated predictions that were subsequently inverse-transformed to the original scale using 10-fold cross-validation. For classification, models were trained directly on categorical targets and produced class probability predictions using 10-fold cross-validation.

**Performance Evaluation**: Regression model performance was assessed using multiple metrics including mean squared error (MSE), mean absolute error (MAE), R-squared, and Pearson correlation coefficient, all calculated using original-scale predictions to ensure interpretability. Classification model performance was evaluated using accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC).

**Model Interpretability**: SHAP (SHapley Additive exPlanations) values were computed for model interpretability using a subset of test samples (maximum 50 samples). SHAP analysis employed a permutation-based explainer algorithm to provide feature-level importance scores and explanations for individual predictions, enabling identification of the most influential features driving model predictions. The SHAP computation utilized the original test data with the same preprocessing pipeline applied during model training, ensuring consistency between model predictions and interpretability analysis.


## Project Structure

```
├── data_loader.py       # Data loading and preprocessing utilities
├── feature_selection.py # Ensemble feature selection implementation
├── model_training.py    # Model training and evaluation logic
├── main.py              # Main execution script for training
├── prediction.py        # Module for making predictions with trained models
├── shap_analysis.py     # Module for computing SHAP values from saved models
├── stacking.py          # Stacking of cross-validation predictions
├── stacking_utils.py    # Helper functions for stacking
├── test_script.py       # Generates simulated demo data and runs the pipeline end to end
├── requirements.txt     # Package dependencies (minimum versions)
├── requirements-lock.txt # Exact package versions used for testing
├── LICENSE              # MIT license for the pipeline code
├── LICENSE-TabPFN       # Prior Labs license covering TabPFN software and model weights
└── README.md            # Project documentation
```

## Features

- **Ensemble Feature Selection**: Combines multiple methods (Mutual Information, XGBoost, CatBoost, Random Forest) to select the most informative features
- **Multi-task Learning**: Supports both regression and classification tasks
- **Cross-Validation Stacking Mode**: Generates information leakage-free predictions for ensemble stacking using K-fold cross-validation
- **GPU Acceleration**: Optional GPU support for XGBoost and CatBoost
- **Covariate Integration**: Optional inclusion of covariates in models
- **Person-level Validation**: Ensures train/test splits respect person identity (no data leakage)
- **TabPFN Models**: Utilizes Transformer-based TabPFN models for accurate tabular data prediction
- **Target Grouping**: Organizes targets into biological categories (longitudinal, pathology, omics, etc.)
- **Predictor Grouping**: Filters predictors by biological or functional groups
- **Prediction Module**: Dedicated module for making predictions with trained models
- **Comprehensive Importance Scores**: Saves importance scores for all proteins/predictors, not just selected ones
- **Selection Status**: Tracks which features were selected for the final model
- **Multiple Methods**: Aggregates importance scores from various feature selection algorithms
- **CSV Export**: Stores feature importance data in easy-to-analyze CSV files
- **SHAP Value Computation**: Calculates and saves SHAP values for model interpretability
- **Post-Training SHAP Analysis**: Dedicated module for computing SHAP values from saved models
- **Model Explainers**: Specialized explainers for TabPFN and traditional models
- **Interpretability Tools**: Support for advanced feature importance analysis
 
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
   To reproduce the tested environment exactly, use the pinned versions instead:
   ```bash
   pip install -r requirements-lock.txt
   ```

3. (Optional, for machines without an NVIDIA GPU) Install the CPU-only PyTorch build
   before the other requirements. It is about 200 MB instead of 6 GB:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

### Installation time and disk space

Installation is dominated by downloading PyTorch and its CUDA libraries. Measured on a
clean virtual environment with an empty pip cache:

| Setup | Download (approx.) | Disk space | Typical time on a desktop with a 100 Mbit/s connection |
|---|---|---|---|
| GPU build (default `pip install torch`) | ~3 GB | ~8 GB | 5-10 minutes |
| CPU-only PyTorch build | ~0.5 GB | ~2 GB | 2-3 minutes |

On a fast data-center connection the GPU build installed in about 2.5 minutes.

On the first run, TabPFN downloads its pretrained model weights (about 75-150 MB per
checkpoint) from Hugging Face and caches them under `~/.cache/tabpfn`. This adds about
a minute the first time and is not repeated.

## Quick Start (Demo with Simulated Data)

The repository does not ship a data file, but `test_script.py` generates a small
simulated dataset in the required input format and runs the full pipeline on it.
This is the fastest way to check that the software works on your machine.

Run a small regression demo (about 10 seconds on a GPU, a few minutes on CPU):

```bash
python test_script.py --data_dir demo_data --output_dir demo_output \
    --n_train_samples 300 --n_test_samples 100 --n_features 50
```

This will:

1. Write simulated input files to `demo_data/` (`predictors.txt.gz`, `targets.txt.gz`,
   `covs.txt.gz`, and the three annotation files). These files follow the format
   described in [Data Structure](#data-structure) and can be used as a template for
   your own data.
2. Select features with CatBoost, train a TabPFN model, and predict on the held-out
   test samples.
3. Write models, predictions, feature importance, and metrics to `demo_output/`
   (see [Output Files](#output-files)).
4. Print `Test results: SUCCESS` together with the R2 and MSE on the test samples.

Other demo variants:

```bash
# Classification target
python test_script.py --data_dir demo_data --output_dir demo_output \
    --n_train_samples 300 --n_test_samples 100 --n_features 50 --classification

# Cross-validation stacking mode (3 folds)
python test_script.py --data_dir demo_data --output_dir demo_output \
    --n_train_samples 300 --n_test_samples 100 --n_features 50 \
    --cv_folds 3 --no_single_split_comparison
```

Running `python test_script.py` with no arguments generates a larger dataset
(12,000 training samples, 3,000 test samples, 1,000 features) to exercise the
TabPFN sampling path for more than 10,000 samples. Add `--comprehensive` to run
all scenarios in sequence.

Note: if the import of TabPFN fails with a message about `SCIPY_ARRAY_API`, set
that variable before running:

```bash
SCIPY_ARRAY_API=1 python test_script.py ...
```

## Data Structure

The framework expects the following data structure in the data directory:

- `predictors.txt.gz`: Predictor variables (e.g., proteins or other omics data)
- `predictor_annotation.txt`: Metadata for predictors including group and type
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

### Predictor Groups
The framework now supports filtering predictors by groups defined in the `predictor_annotation.txt` file. This allows you to train models using specific subsets of predictors based on their biological or functional grouping, such as:

- **proteomics**: Protein abundance measurements
- **metabolomics**: Metabolite measurements
- **genomics**: Genetic markers
- **transcriptomics**: RNA expression data
- And any other custom groups defined in your annotation file

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

Use specific predictor groups for model training:
```bash
python main.py --predictor_group proteomics
```

Use multiple predictor groups:
```bash
python main.py --predictor_group proteomics metabolomics
```

Use all predictor groups:
```bash
python main.py --predictor_group all
```

With custom directories:
```bash
python main.py --data_dir ../data --output_dir ./output --method CatBoost
```

### Cross-Validation Stacking Mode

For generating information leakage-free predictions suitable for ensemble stacking, use the `--cv_folds` parameter:

Basic CV stacking:
```bash
python main.py --cv_folds 5
```

CV stacking with covariates:
```bash
python main.py --cv_folds 10 --include_covariates
```

Process all targets with CV stacking:
```bash
python main.py --cv_folds 5 --target_group all
```

CV stacking with specific predictor groups:
```bash
python main.py --cv_folds 5 --predictor_group proteomics metabolomics
```

**Note:** In CV stacking mode, the `--test_sample_group` parameter is ignored as the framework uses person-level GroupKFold cross-validation to ensure proper data splitting.

### Making Predictions

Basic prediction:
```bash
python prediction.py --models_dir ./models --output_dir ./predictions
```

Force using covariates for all models:
```bash
python prediction.py --force_covariates
```

Predict only specific targets:
```bash
python prediction.py --target_ids target1_id target2_id
```

### Computing SHAP Values for Saved Models

The framework includes a dedicated module for computing SHAP values from saved models.

For a single model:
```bash
python shap_analysis.py --model_path ./models/target123_model.pkl --data_dir ../data
```

For multiple models in a directory:
```bash
python shap_analysis.py --models_dir ./models --data_dir ../data --output_dir ./shap_values
```

For specific target IDs:
```bash
python shap_analysis.py --models_dir ./models --data_dir ../data --target_ids target123 target456
```

Force recomputation of existing SHAP values:
```bash
python shap_analysis.py --model_path ./models/target123_model.pkl --data_dir ../data --force
```

Limit the number of samples for SHAP computation:
```bash
python shap_analysis.py --model_path ./models/target123_model.pkl --data_dir ../data --n_samples 100
```

### Command-line Arguments

#### Training (main.py)
- `--include_covariates`: Include covariates in models
- `--method`: Feature selection method (default: 'CatBoost', options: 'MI', 'XGB', 'CatBoost', 'RF')
- `--targets`: Specific targets to process
- `--target_group`: Target group to process (choices: 'longitudinal', 'pathology', 'omics', 'demographic', 'genetic', 'slope', 'all')
- `--predictor_group`: Predictor groups to use for model building (can specify multiple, use 'all' for all groups)
- `--test_sample_group`: Sample groups to use for testing (ignored in CV mode)
- `--cv_folds`: Number of cross-validation folds for stacking mode (default: None for single train/test split)
- `--data_dir`: Directory containing data files (default: '../data')
- `--output_dir`: Directory for output files (default: '.')
- `--verbose`: Verbosity level (default: 1)
- `--scale_features`: Apply StandardScaler to input features (default: False - no scaling)
- `--save_full_model`: Save model trained on full data
- `--save_train_model`: Save model trained on training data only
- `--compute_shap`: Compute SHAP values for model interpretability

#### Prediction (prediction.py)
- `--data_dir`: Directory containing data files (default: '../data')
- `--models_dir`: Directory containing model files (default: './models')
- `--output_dir`: Directory to save predictions (default: './predictions')
- `--force_covariates`: Force using covariates for all models
- `--force_no_covariates`: Force excluding covariates for all models
- `--force_scaling`: Force applying scaling for all models
- `--force_no_scaling`: Force skipping scaling for all models
- `--no_skip_existing`: Do not skip targets with existing predictions
- `--target_ids`: Specific target IDs to process

#### SHAP Analysis (shap_analysis.py)
- `--model_path`: Path to a specific model file
- `--models_dir`: Directory containing model files
- `--data_dir`: Directory containing data files (default: '../data')
- `--output_dir`: Directory to save SHAP values (defaults to model directory)
- `--n_samples`: Maximum number of samples to use for SHAP computation (default: 50)
- `--target_ids`: Specific target IDs to process
- `--force`: Force recomputation of SHAP values even if they already exist

## Output Files

The framework generates the following outputs:

- `models/`: Directory containing trained models (pickle files)
- `results/`: Directory containing individual target results (JSON)
- `features/`: Directory containing feature importance information (CSV)
- `predictions/`: Directory containing prediction outputs (CSV)
  - `{target_id}_training_predictions.csv.gz`: Training and test set predictions (single split mode)
  - `{target_id}_cv_predictions.csv.gz`: Cross-validation predictions for stacking (CV mode)
  - `{target_id}_full_predictions.csv.gz`: Full dataset predictions
- `shap_values/`: Directory containing SHAP values for model interpretation
- `regression_results.csv`: Summary of regression model performances
- `classification_results.csv`: Summary of classification model performances
- `all_results_summary.csv`: Combined summary of all results

## Prediction Outputs

The framework automatically saves predictions during model training:

- **Training and Test Sets**: Saves predictions for both training and test sets with true values for comparison
- **Full Dataset Models**: Tracks predictions from models trained on the complete dataset
- **Classification Probabilities**: For classification tasks, stores class probabilities
- **Integrated with Model Metadata**: Links prediction files with model metadata

### Accessing Predictions

Prediction outputs are stored in the `predictions` directory:

```
predictions/
  ├── target1_id_training_predictions.csv.gz    # Training and test set predictions
  ├── target1_id_full_predictions.csv.gz       # Full dataset predictions
  ├── target2_id_training_predictions.csv.gz
  └── ...
```

Each predictions file contains:
- `set`: Indicates if the prediction is for 'train', 'test', or 'all' data
- `true_value`: The actual target value
- `prediction`: The model's prediction
- For classification: additional `prob_class_X` columns with class probabilities

## Cross-Validation Predictions for Stacking

When using the `--cv_folds` parameter, the framework generates information leakage-free predictions suitable for ensemble stacking:

```
predictions/
  ├── target1_id_cv_predictions.csv.gz    # CV predictions for stacking
  ├── target2_id_cv_predictions.csv.gz
  └── ...
```

### CV Predictions Format

CV prediction files contain out-of-fold predictions for every sample:

- `sample_id`: Sample identifier
- `fold`: Which CV fold generated this prediction (1 to K)
- `set`: Always 'test' (indicating out-of-fold prediction)
- `true_value`: The actual target value
- `prediction`: The model's prediction from a model that never saw this sample
- For classification: additional `prob_class_X` columns with class probabilities

### Information Leakage Prevention

- Each sample appears exactly once in the CV predictions file
- Predictions are made by models trained on data that excluded that specific sample
- Person-level GroupKFold ensures the same person never appears in both training and test sets within a fold
- Feature selection is performed independently within each fold

### Using CV Predictions for Stacking

The CV predictions are designed for use in ensemble stacking workflows:

1. **Generate CV predictions** for multiple targets or feature sets:
   ```bash
   python main.py --cv_folds 5 --target_group all --predictor_group proteomics
   python main.py --cv_folds 5 --target_group all --predictor_group metabolomics
   ```

2. **Load CV predictions** as features for a second-level model:
   ```python
   import pandas as pd
   
   # Load CV predictions as features
   cv_preds_1 = pd.read_csv('predictions/target1_cv_predictions.csv.gz')
   cv_preds_2 = pd.read_csv('predictions/target2_cv_predictions.csv.gz')
   
   # Combine predictions by sample_id for stacking
   stacking_features = cv_preds_1[['sample_id', 'prediction']].merge(
       cv_preds_2[['sample_id', 'prediction']], on='sample_id', suffixes=['_model1', '_model2']
   )
   ```

3. **Train meta-learner** using the CV predictions as features for your final ensemble model.

## SHAP Value Analysis

The framework computes SHAP (SHapley Additive exPlanations) values to understand feature importance and model decisions:

- **During Training**: SHAP values are computed on test data during model training
- **Post-Training**: SHAP values can be computed for previously saved models using the `shap_analysis.py` module
- **Algorithm Options**: Supports permutation-based and other SHAP algorithms
- **Efficient Sampling**: Limits computation to a manageable number of samples for efficiency

### Accessing SHAP Values

SHAP values are stored in a tidy format in the `shap_values` directory:

```
shap_values/
  ├── target1_id_shap_values.csv.gz    # SHAP values from training
  ├── target1_id_shap_values_post.csv.gz  # SHAP values from post-training analysis
  └── ...
```

Each SHAP file contains:
- `run`: Sample index for which the SHAP value was calculated
- `predictor`: Feature name
- `output`: Class label or target name
- `shap_value`: The SHAP value representing feature contribution

## Feature Selection Details

The feature selection module implements an ensemble approach that combines multiple methods:

- **Mutual Information**: Information-theoretic approach for measuring feature relevance
- **XGBoost**: Feature importance from gradient boosted trees
- **CatBoost**: Feature importance from gradient boosted trees with categorical feature support
- **Random Forest**: Feature importance from random forest ensembles

Features are ranked based on their aggregate importance across all selected methods, and the top k features are selected.

### Accessing Feature Importance

Feature importance information is stored in the `features` directory:

```
features/
  ├── target1_id_feature_importance.csv.gz    # Feature importance for initial model
  ├── target1_id_full_feature_importance.csv.gz  # Feature importance for full-data model
  ├── target2_id_feature_importance.csv.gz
  └── ...
```

Each feature importance file contains:
- `feature_name`: Name of the predictor/protein
- `importance_score`: Relative importance score
- `selected`: Boolean indicating if the feature was selected for the final model

## Predictor Group Filtering

The framework now supports filtering predictors based on their grouping in the annotation file. This is useful for:

1. **Focused Model Building**: Train models using only predictors from specific biological domains
2. **Group Comparison**: Assess prediction performance across different predictor modalities
3. **Resource Optimization**: Reduce computational load by using only relevant predictor subsets
4. **Data Integration**: Incrementally combine predictor groups to evaluate their complementarity

### Predictor Annotation Format

The `predictor_annotation.txt` file should be tab-separated with at minimum these columns:
- `name`: Predictor name (must match column names in predictors.txt.gz)
- `predictor_group`: Group classification (e.g., "proteomics", "metabolomics")
- `type`: Data type ("continuous" or "discrete")

Example:
```
name	predictor_group	type
protein_ABC	proteomics	continuous
gene_XYZ	genomics	continuous
metabolite_123	metabolomics	continuous
```

### Consistency Between Training and Prediction

When using predictor group filtering:
- Models store information about which predictor groups were used for training
- Prediction provides warnings when using different predictor groups than training
- Missing features are zero-filled automatically when needed
- Performance may be impacted if prediction uses different predictor groups than training

## Model Training Details

The framework uses TabPFN (Tabular Prior-Data Fitted Networks) models which leverage transformer architectures pre-trained on thousands of synthetic tabular datasets. These models offer:

- Strong performance on both regression and classification tasks
- Ability to handle categorical features
- Fast training and inference
- Built-in handling of feature interactions

## Performance Metrics

- **Regression**: R², MSE, MAE, Pearson correlation
- **Classification**: Accuracy, Precision, Recall, F1 score, Confusion matrix

**Note**: In CV stacking mode, metrics represent averages across all cross-validation folds, providing a more robust estimate of model performance compared to single train/test splits.

## GPU Support

The framework supports GPU acceleration for feature selection methods that can utilize it (XGBoost, CatBoost) and for TabPFN models when PyTorch CUDA is available.

## Requirements

### Software dependencies

Python packages used by the pipeline (see `requirements.txt` for minimum versions):

- numpy, pandas, scipy
- scikit-learn
- xgboost, catboost (feature selection)
- tabpfn, tabpfn_extensions (TabPFN models and SHAP-based interpretability)
- torch (PyTorch; CUDA build recommended for GPU acceleration)

Install the minimum versions with:
```bash
pip install -r requirements.txt
```

Install the exact versions listed below (recommended for reproducing results) with:
```bash
pip install -r requirements-lock.txt
```

### Tested environment

The pipeline was tested on the following system (September 2026):

| Component | Version |
|---|---|
| Operating system | Ubuntu 22.04.3 LTS (Linux kernel 5.15, x86_64) |
| Python | 3.13.1 |
| numpy | 2.4.2 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| scikit-learn | 1.6.1 |
| xgboost | 3.2.0 |
| catboost | 1.2.10 |
| tabpfn | 8.0.1 |
| tabpfn_extensions | 0.4.1 |
| torch | 2.10.0 (CUDA 12.8 build) |
| GPU | NVIDIA H100 NVL, driver 580.126.09 |

Notes:
- A GPU is optional. Without CUDA, TabPFN, XGBoost, and CatBoost fall back to CPU and run more slowly.
- Other Linux distributions, macOS, and Windows should work with the same Python packages, but were not tested.
- With scikit-learn 1.6 and scipy 1.17, importing TabPFN may require `SCIPY_ARRAY_API=1` in the environment (see [Quick Start](#quick-start-demo-with-simulated-data)).

## License

This repository uses two licenses:

- **Pipeline code** (all `.py` files, notebooks, and documentation in this repository):
  [MIT License](LICENSE). Copyright (c) 2025-2026 Shinya Tasaki.
- **TabPFN software and pretrained model weights** (the `tabpfn` and `tabpfn_extensions`
  packages, the model checkpoints downloaded on first run, and any models or model
  outputs produced with them): [Prior Labs License](LICENSE-TabPFN), which is Apache 2.0
  with an additional attribution provision. TabPFN is developed by
  [Prior Labs](https://priorlabs.ai) and is not part of this repository.

If you distribute models trained with this pipeline, or a product or service built on
them, the Prior Labs License requires that you (A) include a copy of that license and
(B) prominently display "Built with PriorLabs-TabPFN" in the related documentation or
user interface. See Section 10 of [LICENSE-TabPFN](LICENSE-TabPFN) for the exact terms.

Built with PriorLabs-TabPFN.
