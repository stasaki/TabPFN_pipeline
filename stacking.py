"""
Stacking script for combining CV predictions from multiple models
"""
import os
import glob
import pandas as pd
import numpy as np
import argparse
import time
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
import json

from data_loader import load_data
from version import __version__


def load_cv_predictions(predictions_dir, target_ids=None, include_probabilities=True):
    """
    Load CV predictions from multiple models with optimized probability handling
    
    Parameters:
    -----------
    predictions_dir : str
        Directory containing CV prediction files
    target_ids : list, optional
        Specific target IDs to load. If None, loads all available
    include_probabilities : bool, default=True
        Whether to include probability columns for classification models
    prioritize_probabilities : bool, default=True
        Whether to prioritize probabilities over hard predictions as primary features
        
    Returns:
    --------
    dict
        Dictionary with target_id as keys and DataFrames as values
    """
    print(f"Loading CV predictions from {predictions_dir}")
    
    # Find all CV prediction files
    if target_ids:
        cv_files = []
        missing_targets = []
        for target_id in target_ids:
            file_path = os.path.join(predictions_dir, f"{target_id}_stacking_predictions.csv.gz")
            if os.path.exists(file_path):
                cv_files.append(file_path)
            else:
                missing_targets.append(target_id)
        
        if missing_targets:
            print(f"Warning: CV predictions not found for targets: {missing_targets}")
    else:
        cv_files = glob.glob(os.path.join(predictions_dir, "*_stacking_predictions.csv.gz"))
    
    if not cv_files:
        raise ValueError(f"No CV prediction files found in {predictions_dir}")
    
    cv_predictions = {}
    
    for file_path in cv_files:
        target_id = os.path.basename(file_path).replace("_stacking_predictions.csv.gz", "")
        
        try:
            df = pd.read_csv(file_path, compression='gzip', dtype={'sample_id': str})
            
            # Validate required columns
            required_cols = ['sample_id', 'prediction']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {file_path}, skipping")
                continue
            
            cv_df = df.copy()
            
            if len(cv_df) == 0:
                print(f"Warning: No CV predictions found in {file_path}, skipping")
                continue
            
            # Always start with base columns
            keep_cols = ['sample_id']
            if 'true_value' in cv_df.columns:
                keep_cols.append('true_value')
                
            # Handle probability columns for classification
            prob_cols = [col for col in cv_df.columns if col.startswith('prob_class_')]
            
            if include_probabilities and prob_cols:
                # Detect if binary or multi-class classification
                prob_class_numbers = []
                for col in prob_cols:
                    try:
                        # Extract class number from column name like 'prob_class_0', 'prob_class_1'
                        class_num = col.split('prob_class_')[-1]
                        # Handle both numeric and string class labels
                        prob_class_numbers.append(class_num)
                    except:
                        prob_class_numbers.append(col)
                
                unique_classes = sorted(set(prob_class_numbers))
                n_classes = len(unique_classes)
                
                if n_classes == 2:
                    # Binary classification: keep only positive class probability (usually class 1)
                    # First try to find class '1', then try the higher numeric class, then fall back to last alphabetically
                    positive_class_col = None
                    
                    # Strategy 1: Look for '1' or 1
                    for class_num in ['1', 1, '1.0']:
                        candidate_col = f'prob_class_{class_num}'
                        if candidate_col in prob_cols:
                            positive_class_col = candidate_col
                            break
                    
                    # Strategy 2: If no '1', take the highest numeric class
                    if positive_class_col is None:
                        numeric_classes = []
                        for class_num in prob_class_numbers:
                            try:
                                numeric_classes.append((float(class_num), class_num))
                            except:
                                pass
                        if numeric_classes:
                            _, highest_class = max(numeric_classes)
                            positive_class_col = f'prob_class_{highest_class}'
                    
                    # Strategy 3: Fall back to last alphabetically
                    if positive_class_col is None:
                        positive_class_col = f'prob_class_{unique_classes[-1]}'
                    
                    if positive_class_col in prob_cols:
                        selected_prob_cols = [positive_class_col]
                        print(f"  Binary classification: using {positive_class_col} (dropping redundant class 0)")
                    else:
                        # Fallback: use first available probability column
                        selected_prob_cols = [prob_cols[0]]
                        print(f"  Binary classification: using {prob_cols[0]} as fallback")
                        
                else:
                    # Multi-class: keep all probabilities
                    selected_prob_cols = prob_cols
                    print(f"  Multi-class classification: using all {len(prob_cols)} probability columns")
                
                # Add selected probability columns
                keep_cols.extend(selected_prob_cols)
                    
            else:
                # No probabilities available or requested - use hard predictions only
                keep_cols.append('prediction')
                if prob_cols:
                    print(f"  Classification target but probabilities disabled - using hard predictions only")
                else:
                    print(f"  No probability columns found - using hard predictions (likely regression)")
            
            cv_predictions[target_id] = cv_df[keep_cols].copy()
            print(f"Loaded {len(cv_df)} CV predictions for {target_id} with {len(keep_cols)-1} feature columns")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    print(f"Successfully loaded CV predictions for {len(cv_predictions)} targets")
    return cv_predictions


def create_stacking_features(cv_predictions, target_variable, exclude_related=None):
    """
    Create feature matrix from CV predictions, excluding the target variable
    
    Parameters:
    -----------
    cv_predictions : dict
        Dictionary of CV predictions from load_cv_predictions()
    target_variable : str
        Target variable to predict (will be excluded from features)
    exclude_related : list, optional
        Additional target IDs to exclude (e.g., related variables)
        
    Returns:
    --------
    tuple
        (feature_matrix_df, target_series, sample_ids, feature_names)
    """
    if target_variable not in cv_predictions:
        raise ValueError(f"Target variable '{target_variable}' not found in CV predictions")
    
    # Get target values and sample IDs from the target variable
    target_df = cv_predictions[target_variable]
    if 'true_value' not in target_df.columns:
        raise ValueError(f"Target variable '{target_variable}' does not have true_value column")
    
    # Use sample_id and true_value from target
    target_samples = target_df[['sample_id', 'true_value']].drop_duplicates()
    target_samples = target_samples.dropna(subset=['true_value'])  # Remove samples with missing targets
    
    print(f"Target variable: {target_variable}")
    print(f"Samples with valid target values: {len(target_samples)}")
    
    # Prepare exclusion list
    exclude_targets = {target_variable}
    if exclude_related:
        exclude_targets.update(exclude_related)
        print(f"Excluding related targets: {exclude_related}")
    
    # Create feature matrix
    feature_dfs = []
    feature_names = []
    
    for pred_target_id, pred_df in cv_predictions.items():
        if pred_target_id in exclude_targets:
            print(f"Excluding {pred_target_id} (target or related variable)")
            continue
            
        # Get prediction columns (main prediction + any probability columns)
        pred_cols = (['prediction'] if 'prediction' in pred_df.columns else []) + [col for col in pred_df.columns if col.startswith('prob_class_')]
        
        
        # Merge with target samples to keep only samples with valid targets
        merged_df = target_samples[['sample_id']].merge(
            pred_df[['sample_id'] + pred_cols], 
            on='sample_id', 
            how='left'
        )
        
        # Rename columns to include target ID
        for col in pred_cols:
            new_col_name = f"{pred_target_id}_{col}"
            merged_df = merged_df.rename(columns={col: new_col_name})
            feature_names.append(new_col_name)
        
        feature_dfs.append(merged_df[['sample_id'] + [f"{pred_target_id}_{col}" for col in pred_cols]])
    
    if not feature_dfs:
        raise ValueError("No features available after excluding target variable")
    
    # Combine all features
    feature_matrix_df = target_samples[['sample_id']].copy()
    for feat_df in feature_dfs:
        feature_matrix_df = feature_matrix_df.merge(feat_df, on='sample_id', how='left')
    
    # Get final feature matrix and target
    X_df = feature_matrix_df.drop('sample_id', axis=1)
    y = target_samples.set_index('sample_id').loc[feature_matrix_df['sample_id'], 'true_value'].values
    sample_ids = feature_matrix_df['sample_id'].values
    
    print(f"Created feature matrix: {X_df.shape[0]} samples × {X_df.shape[1]} features")
    print(f"Features from {len(feature_dfs)} different models")
    
    # Check for missing values
    missing_stats = X_df.isnull().sum()
    if missing_stats.sum() > 0:
        print("\nMissing value statistics:")
        for col, missing_count in missing_stats[missing_stats > 0].items():
            print(f"  {col}: {missing_count} missing ({missing_count/len(X_df)*100:.1f}%)")
        
        # Fill missing values with median (could be made more sophisticated)
        X_df = X_df.fillna(X_df.median())
        print("Filled missing values with median")
    
    return X_df, y, sample_ids, list(X_df.columns)


def determine_target_type(y):
    """
    Determine if target is regression or classification
    
    Parameters:
    -----------
    y : array-like
        Target values
        
    Returns:
    --------
    str
        'regression' or 'classification'
    """
    unique_vals = np.unique(y[~pd.isnull(y)])
    
    # If any values are strings, it's definitely classification
    if any(isinstance(val, str) for val in unique_vals):
        return 'classification'
    
    # If all values are numeric
    try:
        # Convert to float to handle both int and float inputs
        numeric_vals = [float(val) for val in unique_vals]
        
        # Classification heuristics for numeric data:
        # 1. Small number of unique values (< 10)
        # 2. All values are integers (even if stored as floats)
        if (len(unique_vals) < 10 and 
            all(val.is_integer() for val in numeric_vals)):
            return 'classification'
        else:
            return 'regression'
            
    except (ValueError, TypeError):
        # If conversion to float fails, assume classification
        return 'classification'


def train_stacking_model(X_df, y, sample_ids, target_variable, target_type=None, 
                        cv_folds=5, random_state=42, gpu=True, verbose=1,
                        class_weights=None, auto_class_weights=None, sample_weights=None):
    """
    Train a stacking model using CatBoost with cross-validation
    
    Parameters:
    -----------
    X_df : pandas.DataFrame
        Feature matrix
    y : array-like
        Target values
    sample_ids : array-like
        Sample IDs
    target_variable : str
        Name of target variable
    target_type : str, optional
        'regression' or 'classification'. If None, will be auto-determined
    cv_folds : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    gpu : bool, default=True
        Whether to use GPU
    verbose : int, default=1
        Verbosity level
    class_weights : dict, str, or None, default=None
        Class weights for handling imbalanced datasets. Options:
        - None: No class weighting
        - 'balanced': Automatically balance classes (CatBoost 'Balanced' mode)
        - dict: Manual weights mapping {class_label: weight}
        - 'sqrt_balanced': Square root balanced weighting
    auto_class_weights : str or None, default=None
        CatBoost auto class weights mode. Options:
        - None: No auto weighting
        - 'Balanced': Automatic balanced weighting
        - 'SqrtBalanced': Square root balanced weighting
        Note: This overrides class_weights if specified
    sample_weights : array-like or None, default=None
        Individual sample weights. Must have same length as y.
        
    Returns:
    --------
    dict
        Results dictionary with model, predictions, and metrics
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, cohen_kappa_score,
        r2_score, mean_squared_error
    )
    from catboost import CatBoostRegressor, CatBoostClassifier
    
    if target_type is None:
        target_type = determine_target_type(y)
    
    print(f"\nTraining {target_type} stacking model for {target_variable}")
    print(f"Features: {X_df.shape[1]}, Samples: {X_df.shape[0]}")
    
    # Handle class imbalance for classification
    if target_type == 'classification':
        # Print class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        print(f"Class distribution: {class_distribution}")
        
        # Calculate class imbalance ratio
        max_count = max(class_counts)
        min_count = min(class_counts)
        imbalance_ratio = max_count / min_count
        print(f"Class imbalance ratio: {imbalance_ratio:.2f} (max/min class frequency)")
        
        if imbalance_ratio > 2:
            print("⚠️  Significant class imbalance detected. Consider using class weighting.")
        
        # Process class weights
        processed_class_weights = None
        processed_auto_weights = None
        
        if auto_class_weights is not None:
            processed_auto_weights = auto_class_weights
            print(f"Using auto class weights: {auto_class_weights}")
        elif class_weights is not None:
            if class_weights == 'balanced':
                processed_auto_weights = 'Balanced'
                print("Using balanced class weights (automatic)")
            elif class_weights == 'sqrt_balanced':
                processed_auto_weights = 'SqrtBalanced' 
                print("Using square root balanced class weights (automatic)")
            elif isinstance(class_weights, dict):
                processed_class_weights = class_weights
                print(f"Using manual class weights: {class_weights}")
            else:
                print(f"Warning: Unknown class_weights value: {class_weights}. Ignoring.")
        
        # Validate sample weights
        if sample_weights is not None:
            if len(sample_weights) != len(y):
                raise ValueError(f"sample_weights length ({len(sample_weights)}) must match y length ({len(y)})")
            print(f"Using individual sample weights (min: {np.min(sample_weights):.3f}, "
                  f"max: {np.max(sample_weights):.3f}, mean: {np.mean(sample_weights):.3f})")
    else:
        # For regression, only sample weights are relevant
        processed_class_weights = None
        processed_auto_weights = None
        if sample_weights is not None:
            if len(sample_weights) != len(y):
                raise ValueError(f"sample_weights length ({len(sample_weights)}) must match y length ({len(y)})")
            print(f"Using individual sample weights (min: {np.min(sample_weights):.3f}, "
                  f"max: {np.max(sample_weights):.3f}, mean: {np.mean(sample_weights):.3f})")
        if class_weights is not None or auto_class_weights is not None:
            print("Warning: class_weights and auto_class_weights are ignored for regression tasks.")
    
    # Extract person IDs for GroupKFold (assuming sample_id format: person_measurement)
    person_ids = pd.Series(sample_ids).str.split('_', expand=True)[0]
    
    # Set up cross-validation
    gkf = GroupKFold(n_splits=cv_folds)
    
    # Configure CatBoost parameters
    catboost_params = {
        'random_state': random_state,
        'verbose': 100 if verbose >= 2 else False,
        'iterations': 1000,
        'early_stopping_rounds': 100
    }
    
    # Add class weighting for classification
    if target_type == 'classification':
        if processed_auto_weights is not None:
            catboost_params['auto_class_weights'] = processed_auto_weights
        elif processed_class_weights is not None:
            catboost_params['class_weights'] = processed_class_weights
    
    if gpu:
        catboost_params.update({
            'task_type': 'GPU',
            'devices': '0'
        })
    
    # Initialize model
    if target_type == 'regression':
        model = CatBoostRegressor(**catboost_params)
    else:
        model = CatBoostClassifier(**catboost_params)
    
    # Cross-validation
    cv_predictions = []
    cv_metrics = []
    fold_models = []
    
    X = X_df.values
    
    # Get number of classes for classification
    if target_type == 'classification':
        n_classes = len(np.unique(y))
        print(f"Number of classes: {n_classes}")
    
    def calculate_somers_d(y_true, y_proba):
        """Calculate Somers' D statistic"""
        try:
            auc = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
            return 2 * (auc - 0.5)
        except:
            return np.nan
    
    def calculate_normalized_auprc(y_true, y_proba, average='macro'):
        """Calculate normalized Area Under Precision-Recall Curve"""
        try:
            if average == 'macro':
                pr_auc = average_precision_score(y_true, y_proba, average='macro')
            else:  # weighted
                pr_auc = average_precision_score(y_true, y_proba, average='weighted')
            
            # Normalize by random baseline
            pos_ratio = np.mean(y_true) if len(np.unique(y_true)) == 2 else 1/len(np.unique(y_true))
            normalized_pr_auc = (pr_auc - pos_ratio) / (1 - pos_ratio)
            return max(0, normalized_pr_auc)  # Ensure non-negative
        except:
            return np.nan
    
    def calculate_classification_metrics(y_true, y_pred, y_proba=None):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Macro metrics
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Probability-based metrics (if probabilities available)
        if y_proba is not None:
            try:
                # AUC metrics
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['macro_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['weighted_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    metrics['macro_pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
                    metrics['weighted_pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
                    
                    # Somers' D
                    metrics['macro_somers_d'] = calculate_somers_d(y_true, y_proba[:, 1])
                    metrics['weighted_somers_d'] = metrics['macro_somers_d']  # Same for binary
                    
                    # Normalized AUPRC
                    metrics['macro_n_auprc'] = calculate_normalized_auprc(y_true, y_proba[:, 1], 'macro')
                    metrics['weighted_n_auprc'] = calculate_normalized_auprc(y_true, y_proba[:, 1], 'weighted')
                    
                else:
                    # Multi-class classification
                    metrics['macro_auc'] = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovr')
                    metrics['weighted_auc'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
                    metrics['macro_pr_auc'] = average_precision_score(y_true, y_proba, average='macro')
                    metrics['weighted_pr_auc'] = average_precision_score(y_true, y_proba, average='weighted')
                    
                    # Somers' D
                    metrics['macro_somers_d'] = calculate_somers_d(y_true, y_proba)
                    metrics['weighted_somers_d'] = 2 * (roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr') - 0.5)
                    
                    # Normalized AUPRC
                    metrics['macro_n_auprc'] = calculate_normalized_auprc(y_true, y_proba, 'macro')
                    metrics['weighted_n_auprc'] = calculate_normalized_auprc(y_true, y_proba, 'weighted')
                    
            except Exception as e:
                print(f"Warning: Could not calculate probability-based metrics: {e}")
                # Set missing metrics to NaN
                prob_metrics = ['macro_auc', 'weighted_auc', 'macro_pr_auc', 'weighted_pr_auc',
                               'macro_somers_d', 'weighted_somers_d', 'macro_n_auprc', 'weighted_n_auprc']
                for metric in prob_metrics:
                    metrics[metric] = np.nan
        else:
            # Set probability-based metrics to NaN if no probabilities
            prob_metrics = ['macro_auc', 'weighted_auc', 'macro_pr_auc', 'weighted_pr_auc',
                           'macro_somers_d', 'weighted_somers_d', 'macro_n_auprc', 'weighted_n_auprc']
            for metric in prob_metrics:
                metrics[metric] = np.nan
        
        # Sample and class counts
        metrics['n_samples'] = len(y_true)
        metrics['n_classes'] = len(np.unique(y_true))
        
        return metrics
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=person_ids)):
        print(f"\nTraining fold {fold + 1}/{cv_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Extract sample weights for this fold if provided
        sample_weight_train = sample_weights[train_idx] if sample_weights is not None else None
        sample_weight_val = sample_weights[val_idx] if sample_weights is not None else None
        
        # Train fold model
        if target_type == 'regression':
            fold_model = CatBoostRegressor(**catboost_params)
        else:
            fold_model = CatBoostClassifier(**catboost_params)
        
        # Prepare fit parameters
        fit_params = {
            'X': X_train,
            'y': y_train,
            'eval_set': (X_val, y_val),
            'verbose': verbose >= 2
        }
        
        if sample_weight_train is not None:
            fit_params['sample_weight'] = sample_weight_train
        
        fold_model.fit(**fit_params)
        
        # Make predictions
        val_pred = fold_model.predict(X_val)
        
        # Get probabilities for classification
        if target_type == 'classification':
            try:
                val_proba = fold_model.predict_proba(X_val)
            except:
                val_proba = None
        else:
            val_proba = None
        
        # Store predictions
        for i, idx in enumerate(val_idx):
            pred_entry = {
                'sample_id': sample_ids[idx],
                'fold': fold + 1,
                'true_value': y_val[i],
                'prediction': val_pred[i]
            }
            if val_proba is not None:
                pred_entry['probability'] = val_proba[i].tolist()
            cv_predictions.append(pred_entry)
        
        # Calculate fold metrics
        if target_type == 'regression':
            fold_r2 = r2_score(y_val, val_pred)
            fold_mse = mean_squared_error(y_val, val_pred)
            fold_metrics = {'r2': fold_r2, 'mse': fold_mse}
            cv_metrics.append(fold_metrics)
            print(f"Fold {fold + 1} - R²: {fold_r2:.4f}, MSE: {fold_mse:.4f}")
        else:
            fold_metrics = calculate_classification_metrics(y_val, val_pred, val_proba)
            cv_metrics.append(fold_metrics)
            print(f"Fold {fold + 1} - Accuracy: {fold_metrics['accuracy']:.4f}, "
                  f"Macro F1: {fold_metrics['macro_f1']:.4f}, "
                  f"Weighted F1: {fold_metrics['weighted_f1']:.4f}")
            if not np.isnan(fold_metrics['macro_auc']):
                print(f"  AUC (Macro): {fold_metrics['macro_auc']:.4f}, "
                      f"AUC (Weighted): {fold_metrics['weighted_auc']:.4f}")
        
        fold_models.append(fold_model)
    
    # Train final model on all data
    print("\nTraining final model on all data...")
    final_model = CatBoostRegressor(**catboost_params) if target_type == 'regression' else CatBoostClassifier(**catboost_params)
    
    # Prepare final model fit parameters
    final_fit_params = {
        'X': X,
        'y': y,
        'verbose': verbose >= 2
    }
    
    if sample_weights is not None:
        final_fit_params['sample_weight'] = sample_weights
    
    final_model.fit(**final_fit_params)
    
    # Calculate overall CV metrics
    cv_df = pd.DataFrame(cv_predictions)
    
    def extract_value(val):
        """Extract scalar value from arrays or return as-is"""
        if isinstance(val, np.ndarray):
            return val.item() if val.size == 1 else val[0]
        else:
            return val
    
    if target_type == 'regression':
        overall_r2 = r2_score(cv_df['true_value'], cv_df['prediction'])
        overall_mse = mean_squared_error(cv_df['true_value'], cv_df['prediction'])
        overall_metrics = {'r2': overall_r2, 'mse': overall_mse}
        print(f"\nOverall CV Performance - R²: {overall_r2:.4f}, MSE: {overall_mse:.4f}")
    else:
        cv_df['prediction'] = cv_df['prediction'].apply(extract_value)
        
        # Extract probabilities if available
        if 'probability' in cv_df.columns:
            try:
                proba_matrix = np.array(cv_df['probability'].tolist())
            except:
                proba_matrix = None
        else:
            proba_matrix = None
        
        overall_metrics = calculate_classification_metrics(
            cv_df['true_value'], cv_df['prediction'], proba_matrix
        )
        
        print(f"\nOverall CV Performance:")
        print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
        print(f"  Macro F1: {overall_metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {overall_metrics['weighted_f1']:.4f}")
        print(f"  Cohen's Kappa: {overall_metrics['kappa']:.4f}")
        if not np.isnan(overall_metrics['macro_auc']):
            print(f"  Macro AUC: {overall_metrics['macro_auc']:.4f}")
            print(f"  Weighted AUC: {overall_metrics['weighted_auc']:.4f}")
            print(f"  Macro Somers' D: {overall_metrics['macro_somers_d']:.4f}")
            print(f"  Weighted Somers' D: {overall_metrics['weighted_somers_d']:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_df.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return {
        'final_model': final_model,
        'fold_models': fold_models,
        'cv_predictions': cv_df,
        'overall_metrics': overall_metrics,
        'fold_metrics': cv_metrics,
        'feature_importance': feature_importance,
        'target_type': target_type,
        'target_variable': target_variable,
        'cv_folds': cv_folds,
        'n_features': X_df.shape[1],
        'n_samples': X_df.shape[0],
        'class_weights_config': {
            'class_weights': class_weights,
            'auto_class_weights': auto_class_weights,
            'processed_class_weights': processed_class_weights,
            'processed_auto_weights': processed_auto_weights,
            'sample_weights_provided': sample_weights is not None
        } if target_type == 'classification' else None
    }

def save_stacking_results(results, output_dir, target_variable):
    """
    Save stacking model results
    
    Parameters:
    -----------
    results : dict
        Results from train_stacking_model()
    output_dir : str
        Output directory
    target_variable : str
        Target variable name
    """
    import pickle
    
    # Create stacking directory
    stacking_dir = os.path.join(output_dir, 'stacking')
    os.makedirs(stacking_dir, exist_ok=True)
    
    # Save model
    model_file = os.path.join(stacking_dir, f"{target_variable}_stacking_model.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': results['final_model'],
            'target_variable': target_variable,
            'target_type': results['target_type'],
            'feature_importance': results['feature_importance'],
            'overall_metrics': results['overall_metrics'],
            'n_features': results['n_features'],
            'n_samples': results['n_samples'],
            'cv_folds': results['cv_folds'],
            'framework_version': __version__
        }, f)
    
    # Save CV predictions
    pred_file = os.path.join(stacking_dir, f"{target_variable}_stacking_cv_predictions.csv.gz")
    results['cv_predictions'].to_csv(pred_file, index=False, compression='gzip')
    
    # Save feature importance
    feat_file = os.path.join(stacking_dir, f"{target_variable}_stacking_feature_importance.csv.gz")
    results['feature_importance'].to_csv(feat_file, index=False, compression='gzip')
    
    # Save summary results
    summary = {
        'target_variable': target_variable,
        'target_type': results['target_type'],
        'n_features': results['n_features'],
        'n_samples': results['n_samples'],
        'cv_folds': results['cv_folds'],
        'overall_metrics': results['overall_metrics'],
        'model_file': model_file,
        'predictions_file': pred_file,
        'feature_importance_file': feat_file,
        'framework_version': __version__
    }
    
    summary_file = os.path.join(stacking_dir, f"{target_variable}_stacking_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSaved stacking results:")
    print(f"  Model: {model_file}")
    print(f"  CV Predictions: {pred_file}")
    print(f"  Feature Importance: {feat_file}")
    print(f"  Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Train stacking models using CV predictions')
    parser.add_argument('--predictions_dir', type=str, default='./predictions',
                       help='Directory containing CV prediction files')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for stacking results')
    parser.add_argument('--target_variable', type=str, required=True,
                       help='Target variable to predict with stacking')
    parser.add_argument('--exclude_related', type=str, nargs='+',
                       help='Additional target IDs to exclude from features')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Number of CV folds for stacking model')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"Stacking Model Training")
    print(f"Target variable: {args.target_variable}")
    print(f"Predictions directory: {args.predictions_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"CV folds: {args.cv_folds}")
    print(f"GPU: {args.gpu}")
    
    try:
        # Load CV predictions
        cv_predictions = load_cv_predictions(args.predictions_dir)
        
        # Create stacking features
        X_df, y, sample_ids, feature_names = create_stacking_features(
            cv_predictions, 
            args.target_variable, 
            args.exclude_related
        )
        
        # Train stacking model
        results = train_stacking_model(
            X_df, y, sample_ids, args.target_variable,
            cv_folds=args.cv_folds,
            gpu=args.gpu,
            verbose=args.verbose
        )
        
        # Save results
        save_stacking_results(results, args.output_dir, args.target_variable)
        
        total_time = time.time() - start_time
        print(f"\nStacking model training completed in {total_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()