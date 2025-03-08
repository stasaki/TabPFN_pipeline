# Changelog

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
