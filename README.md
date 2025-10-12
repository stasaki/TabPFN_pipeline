# Biological Data Model Training Framework

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
├── requirements.txt     # Package dependencies
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

Key dependencies include:
- numpy, pandas
- scikit-learn
- xgboost, catboost
- tabpfn
- torch

See requirements.txt for complete dependencies.

## License

[Specify license here]