# Changelog
## [1.7.0] - 2025-03-26

### Added
- SHAP value computation for model interpretability:
  - Added two specialized SHAP explainer functions: TabPFN-specific and default explainer
  - Integrated SHAP value calculation in model training pipeline
  - Implemented storage of SHAP values in both raw and tidy formats
  - Added SHAP value file paths to model metadata
- Enhanced model interpretability:
  - SHAP values for understanding feature importance
  - Support for permutation and partition algorithms
  - Automatic sampling to manage computation time
  - Memory safety checks for high-dimensional data

### Improved
- More comprehensive model metadata with feature importance information
- Better integration with TabPFN extensions for interpretability
- Added safeguards to prevent excessive memory usage for high-dimensional datasets

## [1.6.0] - 2025-03-13

### Added
- Sample group-based test set selection:
  - New `--test_sample_group` parameter for selecting specific sample groups for testing
  - Added sample annotation file support (`sample_annotation.txt`) to define sample groups
  - Integration with train/test split logic to create more realistic evaluation scenarios
  - Storage of test sample group information in model files and result summaries
- TabPFN sample size handling:
  - Automatic detection and handling of datasets exceeding TabPFN's 10,000 sample limit

### Improved
- Enhanced data loading to incorporate sample group information
- Added tracking of group information throughout the pipeline

## [1.5.0] - 2025-03-11

### Added
- Model saving options:
  - New `--save_full_model` parameter to control saving of models trained on full data (default: True)
  - New `--save_train_model` parameter to enable saving of models trained on training data only
  - Added model metadata to distinguish between full-data and training-data models
  - Updated result summaries to track model saving preferences
- Enhanced GPU memory management:
  - Added dedicated clear_gpu_memory() function for explicit memory cleanup
  - Implemented explicit model cleanup and garbage collection after feature selection
  - Added per-method GPU memory clearing to prevent out-of-memory errors

### Improved
- Better memory efficiency for large feature selection tasks
- More explicit control over which models are saved
- Documentation of model types in model files


## [1.4.0] - 2025-03-11

### Added
- Predictor group filtering capability:
  - New `--predictor_group` parameter in main.py for filtering predictors by group during training
  - Integration with data_loader.py to filter predictors based on predictor annotation
  - Updated prediction.py to support the same predictor group filtering
  - Storage of predictor group information in model files and result summaries
- Enhanced predictor annotation handling:
  - Support for predictor_annotation.txt with group and type information
  - Visualization of available predictor groups during execution
  - Warnings when prediction uses different predictor groups than training

### Improved
- Data loading process now handles predictor filtering efficiently
- Model serialization includes predictor group metadata
- Better error handling when predictor groups don't match between training and prediction
- Extended documentation for predictor group usage

## [1.3.0] - 2025-03-08

### Added
- Sample ID tracking in prediction outputs:
  - Include sample IDs in training and testing prediction files
  - Include sample IDs in full dataset prediction files
- Enhanced traceability of model predictions back to original samples

## [1.2.0] - 2025-03-08

### Added
- Feature importance tracking and storage:
  - Save detailed feature importance scores for all proteins/predictors
  - Store feature importance information in separate CSV files
  - Include feature importance file paths in model metadata
- Prediction outputs for model evaluation:
  - Save predicted values for both training and test sets during model training
  - Save predictions from models trained on the full dataset
  - Include probabilities for classification models
  - Link prediction files to model metadata
- Additional result summary information for tracking feature importance and prediction files

### Improved
- Extended output directories to include 'features' and 'predictions'
- Enhanced model serialization to include paths to feature importance and prediction files
- Comprehensive feature importance reporting with selection status and importance scores

## [1.1.0] - 2025-03-07

### Added
- Feature scaling control parameter (`--scale_features`) for training models
- Command-line options for controlling scaling behavior during prediction:
  - `--force_scaling` to enforce feature scaling
  - `--force_no_scaling` to skip feature scaling
- Framework version tracking in model files and results
- Scaling information stored in model metadata for consistent prediction

### Changed
- Feature selection pipeline modified to conditionally apply StandardScaler
- Prediction pipeline enhanced to respect model-specific scaling preferences
- Output files (JSON and CSV) now include scaling configuration
- Updated README with documentation on feature scaling options

### Improved
- More informative logging about scaling status during training and prediction
- Performance transparency by tracking scaling method in results summaries

## [1.0.0] - 2025-03-07
### Added
- Initial release of the biological data model training framework
