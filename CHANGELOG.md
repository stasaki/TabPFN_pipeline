# Changelog

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
