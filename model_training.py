"""
Module for training and evaluating models
"""
import sys
import numpy as np
import pandas as pd
import time
import os
import pickle
import json
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tabpfn import TabPFNRegressor, TabPFNClassifier
from version import __version__
from tabpfn_extensions import interpretability

from feature_selection import create_feature_selection_pipeline
from data_loader import check_existing_results, load_existing_result, get_samples_by_group

# Maximum number of samples TabPFN can handle
TABPFN_MAX_SAMPLES = 10000

MODEL_CLS_PATH = "../../Resources/models/tabpfn-v2-classifier.ckpt"
MODEL_REG_PATH = "../../Resources/models/tabpfn-v2-regressor.ckpt"
MODEL_CLS_PATH = "auto"
MODEL_REG_PATH = "auto"

def compute_and_save_shap_values(model, X_test_sample, feature_names, output_dir, target_id, n_samples=50, is_classifier=True):
    """
    Compute SHAP values for a trained model and save to file.
    Handles both classification (multi-class) and regression tasks.

    Parameters:
    -----------
    model : TabPFNClassifier or TabPFNRegressor or similar model object
        Trained model.
    X_test_sample : numpy.ndarray
        Sample of test data for SHAP value computation (limit to ~50 samples for performance).
    feature_names : list or numpy.ndarray
        Names of features.
    output_dir : str
        Directory to save SHAP values.
    target_id : str
        Target ID for file naming.
    n_samples : int, default=50
        Number of samples to use for SHAP computation.
    is_classifier : bool, default=True
        Whether the model is a classifier or regressor. This helps determine output labels.

    Returns:
    --------
    str or None
        Path to the saved SHAP values file, or None if an error occurred.
    """
    # Limit number of samples to compute SHAP values for
    if X_test_sample.shape[0] > n_samples:
        # Randomly select n_samples from X_test_sample
        idx = np.random.choice(X_test_sample.shape[0], n_samples, replace=False)
        X_shap = X_test_sample[idx]
    else:
        X_shap = X_test_sample

    print(f"Computing SHAP values for {X_shap.shape[0]} samples...")

    # Calculate the number of features
    n_features = X_shap.shape[1]

    # Calculate required max_evals for the permutation explainer
    # Note: Some SHAP algorithms might need different settings
    required_evals = 2 * n_features + 1
    print(f"Required evaluations (for permutation): {required_evals}")

    try:
        # Algorithm can be 'permutation' or 'partition', 'kernel', etc. depending on library/model
        # Ensure the algorithm and parameters are suitable for your specific 'interpretability.shap' source
        shap_values = interpretability.shap.get_shap_values(
            estimator=model,
            test_x=X_shap,
            attribute_names=feature_names,
            algorithm="permutation", # Verify this is the correct algorithm name
            max_evals=required_evals
        )

        # --- Handle different SHAP value array shapes ---
        shap_array = shap_values.values
        shape = shap_array.shape
        predictor_names = shap_values.feature_names # Get predictor names

        if len(shape) == 3:
            # Shape is (runs/samples, num_predictors, num_outputs) - Multi-class classification
            runs, num_predictors, num_outputs = shape
            if is_classifier:
                output_labels = [f"class_{i}" for i in range(num_outputs)]
            else:
                # If user flagged as regressor but got 3D, treat as multi-output regression
                print("Warning: SHAP values have 3 dimensions, but is_classifier=False. Assuming multi-output target.")
                output_labels = [f"output_{i}" for i in range(num_outputs)]

        elif len(shape) == 2:
            # Shape is (runs/samples, num_predictors) - Regression or Binary Classification (one output)
            runs, num_predictors = shape
            num_outputs = 1 # Only one output dimension
            if is_classifier:
                 # If user flagged as classifier but got 2D, assume binary output (e.g., prob of class 1)
                 print("Warning: SHAP values have 2 dimensions, but is_classifier=True. Assuming SHAP for positive class.")
                 # You might want a more specific label if possible, e.g., model.classes_[1]
                 output_labels = ["class_1_prob"] # Or a more generic term
            else:
                 # Regression case
                 output_labels = ["target_value"]
        else:
            # Handle unexpected shapes
            raise ValueError(f"Unexpected SHAP values array shape: {shape}. Expected 2 or 3 dimensions.")

        # --- Create tidy DataFrame ---
        # Ensure variable names match DataFrame columns
        run_ids = np.repeat(np.arange(runs), num_predictors * num_outputs)
        predictor_col = np.tile(np.repeat(predictor_names, num_outputs), runs)
        output_col = np.tile(np.array(output_labels), runs * num_predictors)
        shap_values_flat = shap_array.reshape(-1) # Flatten the array for the column

        # Build the tidy DataFrame
        df_tidy = pd.DataFrame({
            "run": run_ids,          # Corresponds to the sample index in X_shap
            "predictor": predictor_col, # Feature name
            "output": output_col,    # Class label or target name
            "shap_value": shap_values_flat # The SHAP value itself
        })

        # Create directory for SHAP values
        shap_dir = os.path.join(output_dir, 'shap_values')
        os.makedirs(shap_dir, exist_ok=True)

        # Save to gzipped CSV for efficient storage and easy reading
        shap_file = os.path.join(shap_dir, f"{target_id}_shap_values.csv.gz")
        df_tidy.to_csv(shap_file, index=False, compression="gzip")

        print(f"Saved SHAP values to {shap_file} (gzip compressed)")

        return shap_file

    except Exception as e:
        print(f"Error computing or processing SHAP values: {e}")
        # Optionally add more detailed error logging or traceback
        import traceback
        traceback.print_exc()
        return None


def process_target(data, target_name, include_covariates, selected_k, method="CatBoost", 
               verbose=1, gpu=True, output_dir='.', skip_scaling=False, 
               save_full_model=False, save_train_model=False, compute_shap=False):
    """
    Process a single target
    
    Parameters:
    -----------
    data : dict
        Dictionary containing data
    target_name : str
        Name of the target
    include_covariates : bool
        Whether to include covariates
    selected_k : int
        Number of features to select
    method : str, default="CatBoost"
        Feature selection method
    verbose : int, default=1
        Verbosity level
    gpu : bool, default=True
        Whether to use GPU
    output_dir : str, default='.'
        Output directory
    skip_scaling : bool, default=False
        If True, skips the StandardScaler step for input features, assuming data is already scaled
    save_full_model : bool, default=True
        Whether to save the model trained on full data
    save_train_model : bool, default=False
        Whether to save the model trained on training data only
    compute_shap : bool, default=False
        Whether to compute SHAP values for model interpretability
    
    Returns:
    --------
    dict or None
        Result dictionary or None if skipped
    """
    target_start = time.time()
        
    # Check if target exists in the data
    if target_name not in data['Y_df'].columns:
        print(f"Warning: Target {target_name} not found in Y data. Skipping.")
        return None

    try:
        # Get target info from Y_annot_df
        target_info = data['Y_annot_df'][data['Y_annot_df']['name'] == target_name]
        if target_info.empty:
            print(f"Warning: Target {target_name} not found in annotation file. Skipping.")
            return None
            
        # Extract target details including target_id
        target_type = target_info['type'].values[0]
        target_nlevels = None if pd.isna(target_info['nlevels'].values[0]) else int(target_info['nlevels'].values[0])
        target_id = target_info['target_id'].values[0]  # Get target_id for file naming
        
        print(f"Target: {target_name}, ID: {target_id}, Type: {target_type}" + 
              (f", levels: {target_nlevels}" if target_nlevels else "") +
              (f", using pre-scaled data" if skip_scaling else ", scaling data") +
              (f", with SHAP values" if compute_shap else ", without SHAP values"))
        
        # Check if results already exist for this target
        if check_existing_results(target_id, output_dir):
            # Load existing results and return
            existing_result = load_existing_result(target_id, output_dir)
            if existing_result:
                print(f"Added existing result for {target_id} to results summary.")
                return existing_result
            return None

        # Remove rows with NA in the current target column
        valid_idx = data['Y_df'][target_name].notna()
        y = data['Y_df'][target_name][valid_idx].values
        X_valid = data['X'][valid_idx, :]
        covs_valid = data['Covs'][valid_idx, :] if include_covariates else np.zeros((sum(valid_idx), 0))
        sample_id_valid = data['sample_id'][valid_idx].reset_index(drop=True)
        
        # Extract person_id from the sample_id (assuming format is "person_id_measurement")
        sample_id_valid = sample_id_valid.astype(str)  # Convert to string
        person_ids = sample_id_valid.str.split('_', expand=True)[0]  # Get the first part before underscore
        
        # Number of samples for this target
        num_samples = len(y)
        print(f"Number of valid samples: {num_samples}")
        
        if num_samples < 50:
            print(f"Warning: Only {num_samples} samples for target {target_name}. Skipping due to insufficient data.")
            return None
        
        # Determine split method based on test_sample_groups
        if 'test_sample_groups' in data and data['test_sample_groups'] and 'sample_annot_df' in data and not data['sample_annot_df'].empty:
            print(f"Using sample group-based split with groups: {', '.join(data['test_sample_groups'])}")
            
            # Get mask for samples in the specified test groups
            test_mask = get_samples_by_group(sample_id_valid, data['sample_annot_df'], data['test_sample_groups'])
            
            # If no test samples were found in the specified groups, fall back to GroupShuffleSplit
            if test_mask is None or sum(test_mask) == 0:
                print("Warning: No test samples found in the specified groups. Exiting program.")
                sys.exit(1)  # Exit the program with a non-zero exit code indicating an error
            else:
                # Use the test_mask to determine train/test indices
                test_idx = np.where(test_mask)[0]
                train_idx = np.where(~test_mask)[0]
                print(f"Sample group split: {len(train_idx)} train samples, {len(test_idx)} test samples")
        else:
            # Use GroupShuffleSplit to ensure the same person isn't in both train and test
            print("Using person ID-based split")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            train_idx, test_idx = next(gss.split(X_valid, y, groups=person_ids))
        
        # Split the data using these indices
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        covs_train, covs_test = covs_valid[train_idx], covs_valid[test_idx] if include_covariates else (np.zeros((len(train_idx), 0)), np.zeros((len(test_idx), 0)))
        sample_id_train, sample_id_test = sample_id_valid.iloc[train_idx], sample_id_valid.iloc[test_idx]
        person_ids_train, person_ids_test = person_ids.iloc[train_idx], person_ids.iloc[test_idx]
        
        # Apply TabPFN sampling limit for training data if needed
        train_sampling_info = None
        if len(X_train) > TABPFN_MAX_SAMPLES:
            print(f"Training set exceeds TabPFN limit ({len(X_train)} > {TABPFN_MAX_SAMPLES}). Sampling...")
            
            # For classification, use stratified sampling to preserve class distribution
            stratify = y_train if target_type == 'discrete' and len(np.unique(y_train)) < 10 else None
            
            # Sample indices
            indices = np.arange(len(X_train))
            _, sampled_indices = train_test_split(
                indices, 
                test_size=TABPFN_MAX_SAMPLES,
                random_state=42,
                stratify=stratify
            )
            
            # Keep track of original training data size
            train_sampling_info = {
                "original_size": len(X_train),
                "sampled_size": TABPFN_MAX_SAMPLES,
                "sampling_ratio": TABPFN_MAX_SAMPLES / len(X_train),
            }
            
            # Apply sampling
            X_train_sampled = X_train[sampled_indices]
            y_train_sampled = y_train[sampled_indices]
            covs_train_sampled = covs_train[sampled_indices] if include_covariates else np.zeros((TABPFN_MAX_SAMPLES, 0))
            sample_id_train_sampled = sample_id_train.iloc[sampled_indices].reset_index(drop=True)
            
            print(f"Sampled {TABPFN_MAX_SAMPLES} out of {len(X_train)} training samples")
            
            # Replace original data with sampled data
            X_train = X_train_sampled
            y_train = y_train_sampled
            covs_train = covs_train_sampled
            sample_id_train = sample_id_train_sampled
        
        # Apply TabPFN sampling limit for test data if needed
        test_sampling_info = None
        if len(X_test) > TABPFN_MAX_SAMPLES:
            print(f"Test set exceeds TabPFN limit ({len(X_test)} > {TABPFN_MAX_SAMPLES}). Sampling...")
            
            # For classification, use stratified sampling to preserve class distribution
            stratify = y_test if target_type == 'discrete' and len(np.unique(y_test)) < 10 else None
            
            # Sample indices
            indices = np.arange(len(X_test))
            _, sampled_indices = train_test_split(
                indices, 
                test_size=TABPFN_MAX_SAMPLES,
                random_state=42,
                stratify=stratify
            )
            
            # Keep track of original test data size
            test_sampling_info = {
                "original_size": len(X_test),
                "sampled_size": TABPFN_MAX_SAMPLES,
                "sampling_ratio": TABPFN_MAX_SAMPLES / len(X_test),
            }
            
            # Apply sampling
            X_test_sampled = X_test[sampled_indices]
            y_test_sampled = y_test[sampled_indices]
            covs_test_sampled = covs_test[sampled_indices] if include_covariates else np.zeros((TABPFN_MAX_SAMPLES, 0))
            sample_id_test_sampled = sample_id_test.iloc[sampled_indices].reset_index(drop=True)
            
            print(f"Sampled {TABPFN_MAX_SAMPLES} out of {len(X_test)} test samples")
            
            # Replace original data with sampled data
            X_test = X_test_sampled
            y_test = y_test_sampled
            covs_test = covs_test_sampled
            sample_id_test = sample_id_test_sampled
        
        # Verify the split respects person_ids
        train_people = set(person_ids_train)
        test_people = set(person_ids_test)
        overlap = train_people.intersection(test_people)
        print(f"Number of people in both train and test: {len(overlap)}")  # Should be 0
        
        # Determine categorical features based on Covs_annot_df
        categorical_indices = []
        
        # Identify which columns in the covariates are categorical
        cov_categorical_indices = data['Covs_annot_df'].index[data['Covs_annot_df']['type'] == 'discrete'].tolist() if include_covariates else []
        
        # Process differently based on target type
        if target_type == 'continuous':
            # REGRESSION WORKFLOW
            result_dict = train_regression_model(
                X_train, X_test, y_train, y_test,
                covs_train, covs_test,
                sample_id_train, sample_id_test,
                selected_k, data['predictor_names'], data['covariate_names'],
                cov_categorical_indices, categorical_indices,
                target_name, target_id, include_covariates,
                method, verbose, gpu, output_dir, skip_scaling,
                data,  # Pass the full data dictionary to access annotations if needed
                save_full_model, save_train_model, compute_shap  # Pass the model saving and SHAP computation options
            )
            
        elif target_type == 'discrete':
            # CLASSIFICATION WORKFLOW
            result_dict = train_classification_model(
                X_train, X_test, y_train, y_test,
                covs_train, covs_test,
                sample_id_train, sample_id_test,
                selected_k, data['predictor_names'], data['covariate_names'],
                cov_categorical_indices, categorical_indices,
                target_name, target_id, target_nlevels, include_covariates,
                method, verbose, gpu, output_dir, skip_scaling,
                data,  # Pass the full data dictionary to access annotations if needed
                save_full_model, save_train_model, compute_shap  # Pass the model saving and SHAP computation options
            )
            
        else:
            print(f"Warning: Unknown target type '{target_type}' for {target_name}. Skipping.")
            return None
        
        # Calculate elapsed time for this target
        target_elapsed = time.time() - target_start
        print(f"Target processed in {target_elapsed:.2f} seconds.")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"Cleared GPU cache after target {target_name}")
            
        return result_dict
        
    except Exception as e:
        print(f"Error processing target {target_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
        

def sample_data_for_tabpfn(X, y, sample_ids=None, max_samples=TABPFN_MAX_SAMPLES, random_state=42):
    """
    Sample data if it exceeds TabPFN's limit
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target values
    sample_ids : pandas.Series or None
        Sample IDs corresponding to X and y
    max_samples : int, default=TABPFN_MAX_SAMPLES
        Maximum number of samples TabPFN can handle
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        Sampled X, y, and optionally sample_ids
    """
    if len(X) <= max_samples:
        # No sampling needed
        if sample_ids is not None:
            return X, y, sample_ids
        return X, y
    
    print(f"Sampling {max_samples} out of {len(X)} samples for TabPFN (which has a {max_samples} sample limit)")
    
    # Perform sampling
    if sample_ids is not None:
        # If we have sample_ids, sample from indices and return corresponding sample_ids
        indices = np.arange(len(X))
        sampled_indices = np.sort(np.random.RandomState(random_state).choice(indices, size=max_samples, replace=False))
        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]
        sample_ids_sampled = sample_ids.iloc[sampled_indices].reset_index(drop=True)
        return X_sampled, y_sampled, sample_ids_sampled
    else:
        # If no sample_ids, sample directly from X and y
        X_sampled, y_sampled = train_test_split(
            X, y, train_size=max_samples, 
            random_state=random_state, stratify=y if len(np.unique(y)) < 10 else None
        )
        return X_sampled, y_sampled
        

def train_regression_model(X_train, X_test, y_train, y_test, 
                       covs_train, covs_test,
                       sample_id_train, sample_id_test,
                       selected_k, predictor_names, covariate_names,
                       cov_categorical_indices, categorical_indices,
                       target_name, target_id, include_covariates,
                       method="CatBoost", verbose=1, gpu=True, output_dir='.', skip_scaling=False,
                       data=None, save_full_model=False, save_train_model=False, compute_shap=False):
    """
    Train a regression model
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and test feature matrices
    y_train, y_test : numpy.ndarray
        Training and test target values
    covs_train, covs_test : numpy.ndarray
        Training and test covariate matrices
    sample_id_train, sample_id_test : pandas.Series
        Sample IDs for training and test data
    selected_k : int
        Number of features to select
    predictor_names, covariate_names : numpy.ndarray
        Names of predictors and covariates
    cov_categorical_indices, categorical_indices : list
        Indices of categorical features
    target_name : str
        Name of the target
    target_id : str
        ID of the target
    include_covariates : bool
        Whether to include covariates
    method : str
        Feature selection method
    verbose : int
        Verbosity level
    gpu : bool
        Whether to use GPU
    output_dir : str
        Output directory
    skip_scaling : bool
        If True, skips the StandardScaler step
    data : dict or None
        Full data dictionary (optional)
    save_full_model : bool
        Whether to save the model trained on full data
    save_train_model : bool
        Whether to save the model trained on training data only
    compute_shap : bool
        Whether to compute SHAP values for model interpretability
    
    Returns:
    --------
    dict
        Result dictionary
    """
    target_start = time.time()


                
    # Define file paths based on model saving options
    if save_train_model:
        train_model_filename = os.path.join(output_dir, 'models', f"{target_id}_train_model.pkl")
    else:
        train_model_filename = None
        
    if save_full_model:
        full_model_filename = os.path.join(output_dir, 'models', f"{target_id}_model.pkl")
    else:
        full_model_filename = None
    
    # The main model file for results will be either the full model or the train model
    if save_full_model:
        model_filename = full_model_filename
    else:
        model_filename = train_model_filename

    result_filename = os.path.join(output_dir, 'results', f"{target_id}_results.txt")
    
    # Create directory for predictions
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Scale y: create a separate scaler for the target variable
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    
    # Create a pipeline for X that scales and selects features
    pipeline_fs = create_feature_selection_pipeline(
        selected_k=selected_k, 
        methods=[method],
        verbose=verbose,
        gpu=gpu,
        task='regression',
        skip_scaling=skip_scaling
    )
    
    # Fit the pipeline on X_train
    pipeline_fs.fit(X_train, y_train_scaled, feature_selection__feature_names=predictor_names)
    
    # Transform both training and test sets
    X_train_selected = pipeline_fs.transform(X_train)
    X_test_selected = pipeline_fs.transform(X_test)
    
    # Save feature importance after fitting
    feature_importance_file = pipeline_fs.named_steps['feature_selection'].save_feature_importance(output_dir, target_id)
    
    # Get the indices of selected features
    selected_feature_indices = pipeline_fs.named_steps['feature_selection'].get_support(indices=True)
    
    # Get the names of selected features
    selected_feature_names = predictor_names[selected_feature_indices]
    
    # Concatenate the selected features from X with the covariates
    X_train_final = np.hstack([X_train_selected, covs_train]) if include_covariates else X_train_selected
    X_test_final = np.hstack([X_test_selected, covs_test]) if include_covariates else X_test_selected
    
    # Update categorical indices for the final feature matrix
    final_categorical_indices = categorical_indices.copy()
    if include_covariates:
        for cov_idx in cov_categorical_indices:
            final_categorical_indices.append(X_train_selected.shape[1] + cov_idx)
            
    # Initialize and train the regressor on the final training features
    device = "cuda:3" if gpu and torch.cuda.is_available() else "cpu"
    regressor = TabPFNRegressor(
        device=device,
        categorical_features_indices=final_categorical_indices,
        model_path = MODEL_REG_PATH,
        n_estimators=8  # You can adjust this parameter as needed
    )
    regressor.fit(X_train_final, y_train_scaled)
    
    # Feature types list depends on whether covariates are included
    feature_types = ['predictor'] * len(selected_feature_names)
    feature_names = selected_feature_names
    
    if include_covariates:
        feature_types.extend(['covariate'] * len(covariate_names))
        feature_names = np.concatenate([selected_feature_names, covariate_names])
    
    # Predict on both training and test sets (predictions are in scaled space)
    train_predictions_scaled = regressor.predict(X_train_final)
    test_predictions_scaled = regressor.predict(X_test_final)
    
    # Inverse-transform predictions back to the original y scale
    train_predictions = y_scaler.inverse_transform(train_predictions_scaled.reshape(-1, 1)).ravel()
    test_predictions = y_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).ravel()
    
    # Save training and test set predictions with sample IDs
    train_pred_df = pd.DataFrame({
        'sample_id': sample_id_train,  # Add sample IDs
        'set': ['train'] * len(train_predictions),
        'true_value': y_train,
        'prediction': train_predictions
    })
    
    test_pred_df = pd.DataFrame({
        'sample_id': sample_id_test,  # Add sample IDs
        'set': ['test'] * len(test_predictions),
        'true_value': y_test,
        'prediction': test_predictions
    })
    
    # Combine the dataframes
    all_pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
    
    # Save predictions to file
    predictions_file = os.path.join(predictions_dir, f"{target_id}_training_predictions.csv.gz")
    all_pred_df.to_csv(predictions_file, index=False, compression='gzip')
    print(f"Saved training and test predictions to {predictions_file} (gzip compressed)")
    
    # Calculate performance metrics on the original y scale using test set
    mse = mean_squared_error(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    # Calculate Pearson's correlation coefficient
    pearson_r = np.corrcoef(y_test, test_predictions)[0, 1]
    
    # Initialize shap_file variable
    shap_file = None
    
    # Compute SHAP values if requested
    if compute_shap:
        try:
            shap_file = compute_and_save_shap_values(
                regressor, 
                X_test_final, 
                feature_names, 
                output_dir, 
                target_id, 
                n_samples=min(50, X_test_final.shape[0]),
                is_classifier=False
            )
            print(f"Computed and saved SHAP values to {shap_file}")
        except Exception as e:
            print(f"Warning: Could not compute SHAP values: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping SHAP value computation as requested")
        
    # If requested, save the training model
    if save_train_model:       
        with open(train_model_filename, 'wb') as f:
            pickle.dump({
                'model': regressor, 
                'y_scaler': y_scaler,
                'target_id': target_id,
                'target_type': 'continuous',
                'include_covariates': include_covariates,
                'scale_features': not skip_scaling,
                'compute_shap': compute_shap,
                'framework_version': __version__,
                'feature_info': {
                    'feature_name': feature_names,
                    'feature_type': feature_types,
                    'categorical_indices': final_categorical_indices},
                'performance': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'pearson': pearson_r
                },
                'feature_importance_file': feature_importance_file,
                'predictions_file': predictions_file,
                'is_full_data_model': False,
                'is_train_data_model': True,
                'shap_values_file': shap_file
            }, f)
        print(f"Saved regression model (trained on TRAINING data) to {train_model_filename}")
    
    # Create a dictionary for this target's results
    result_dict = {
        "Target": target_name,
        "Target ID": target_id,
        "Type": "Regression",
        "Framework_Version": __version__,
        "Number of samples": len(y_train) + len(y_test),
        "Include Covariates": include_covariates,
        "Scale Features": not skip_scaling,
        "Compute SHAP": compute_shap,
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson_r,
        "Time (s)": time.time() - target_start,
        "Model file": model_filename,
        "Feature importance file": feature_importance_file,
        "Predictions file": predictions_file
    }
    
    # Add SHAP file to result dictionary if it was computed
    if shap_file:
        result_dict["SHAP values file"] = shap_file

    # Only retrain the model using ALL data if requested
    if save_full_model and (len(X_train) + len(X_test)) <= 10000:
        print("Retraining the regression model using all available data...")

        # Combine train and test data
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        covs_all = np.vstack([covs_train, covs_test]) if include_covariates else np.zeros((len(y_all), 0))
        sample_id_all = pd.concat([sample_id_train, sample_id_test]).reset_index(drop=True)  # Combine sample IDs

        # Scale all targets together
        y_all_scaler = StandardScaler()
        y_all_scaled = y_all_scaler.fit_transform(y_all.reshape(-1, 1)).ravel()

        # Refit the feature selection pipeline on all data
        pipeline_fs.fit(X_all, y_all_scaled, feature_selection__feature_names=predictor_names)
        X_all_selected = pipeline_fs.transform(X_all)

        # Save feature importance for all-data model
        all_feature_importance_file = pipeline_fs.named_steps['feature_selection'].save_feature_importance(output_dir, f"{target_id}_full")

        # Get the indices of selected features from all data
        all_selected_feature_indices = pipeline_fs.named_steps['feature_selection'].get_support(indices=True)

        # Get the names of selected features
        all_selected_feature_names = predictor_names[all_selected_feature_indices]

        print(f"Selected features using all data: {all_selected_feature_names}")

        # Prepare final feature matrix
        X_all_final = np.hstack([X_all_selected, covs_all]) if include_covariates else X_all_selected

        # Update categorical indices for the all-data feature matrix
        all_final_categorical_indices = categorical_indices.copy()
        if include_covariates:
            for cov_idx in cov_categorical_indices:
                all_final_categorical_indices.append(X_all_selected.shape[1] + cov_idx)

        # Initialize and train the final regressor on all data
        final_regressor = TabPFNRegressor(
            device=device,
            categorical_features_indices=all_final_categorical_indices,
            model_path = MODEL_REG_PATH,
            n_estimators=8
        )
        final_regressor.fit(X_all_final, y_all_scaled)

        # Get predictions from the final model on all data
        all_predictions_scaled = final_regressor.predict(X_all_final)
        all_predictions = y_all_scaler.inverse_transform(all_predictions_scaled.reshape(-1, 1)).ravel()

        # Save all-data predictions with sample IDs
        all_data_pred_df = pd.DataFrame({
            'sample_id': sample_id_all,  # Add sample IDs
            'set': ['all'] * len(all_predictions),
            'true_value': y_all,
            'prediction': all_predictions
        })

        all_predictions_file = os.path.join(predictions_dir, f"{target_id}_full_predictions.csv.gz")
        all_data_pred_df.to_csv(all_predictions_file, index=False, compression='gzip')
        print(f"Saved full data predictions to {all_predictions_file} (gzip compressed)")

        # Initialize all_shap_file variable
        all_shap_file = None
        
        # Compute SHAP values for full model if requested
        if compute_shap:
            try:
                all_shap_file = compute_and_save_shap_values(
                    final_regressor, 
                    X_all_final, 
                    all_selected_feature_names if not include_covariates else 
                    np.concatenate([all_selected_feature_names, covariate_names]), 
                    output_dir, 
                    f"{target_id}_full", 
                    n_samples=min(50, X_all_final.shape[0]),
                    is_classifier=False
                )
                print(f"Computed and saved SHAP values for full model to {all_shap_file}")
            except Exception as e:
                print(f"Warning: Could not compute SHAP values for full model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping SHAP value computation for full model as requested")

        # Update feature information
        all_feature_types = ['predictor'] * len(all_selected_feature_names)
        all_feature_names = all_selected_feature_names

        if include_covariates:
            all_feature_types.extend(['covariate'] * len(covariate_names))
            all_feature_names = np.concatenate([all_selected_feature_names, covariate_names])

        # Save the ALL-DATA trained model    
        with open(full_model_filename, 'wb') as f:
            pickle.dump({
                'model': final_regressor, 
                'y_scaler': y_all_scaler,
                'target_id': target_id,
                'target_type': 'continuous',
                'include_covariates': include_covariates,
                'scale_features': not skip_scaling,
                'compute_shap': compute_shap,
                'framework_version': __version__,
                'feature_info': {
                    'feature_name': all_feature_names,
                    'feature_type': all_feature_types,
                    'categorical_indices': all_final_categorical_indices},
                'performance': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'pearson': pearson_r
                },
                'feature_importance_file': feature_importance_file,
                'all_feature_importance_file': all_feature_importance_file,
                'predictions_file': predictions_file,
                'all_predictions_file': all_predictions_file,
                'is_full_data_model': True,
                'is_train_data_model': False,
                'shap_values_file': shap_file,
                'all_shap_values_file': all_shap_file
            }, f)
        print(f"Saved regression model (trained on ALL data) to {model_filename}")
        # Update the result dictionary with full-data model info
        result_dict.update({
            "All Feature importance file": all_feature_importance_file,
            "All Predictions file": all_predictions_file
        })
        
        # Add SHAP file to result dictionary if it was computed
        if all_shap_file:
            result_dict["All SHAP values file"] = all_shap_file
    
    # Save the result_dict to an individual text file
    with open(result_filename, 'w') as f:
        # Format the output as pretty JSON
        json.dump(result_dict, f, indent=4)
    print(f"Saved individual result to {result_filename}")
    
    return result_dict


def train_classification_model(X_train, X_test, y_train, y_test, 
                          covs_train, covs_test,
                          sample_id_train, sample_id_test,
                          selected_k, predictor_names, covariate_names,
                          cov_categorical_indices, categorical_indices,
                          target_name, target_id, target_nlevels, include_covariates,
                          method="CatBoost", verbose=1, gpu=True, output_dir='.', skip_scaling=False,
                          data=None, save_full_model=False, save_train_model=False, compute_shap=False):
    """
    Train a classification model
    
    Parameters:
    -----------
    X_train, X_test : numpy.ndarray
        Training and test feature matrices
    y_train, y_test : numpy.ndarray
        Training and test target values
    covs_train, covs_test : numpy.ndarray
        Training and test covariate matrices
    sample_id_train, sample_id_test : pandas.Series
        Sample IDs for training and test data
    selected_k : int
        Number of features to select
    predictor_names, covariate_names : numpy.ndarray
        Names of predictors and covariates
    cov_categorical_indices, categorical_indices : list
        Indices of categorical features
    target_name : str
        Name of the target
    target_id : str
        ID of the target
    target_nlevels : int or None
        Number of levels in the target variable
    include_covariates : bool
        Whether to include covariates
    method : str
        Feature selection method
    verbose : int
        Verbosity level
    gpu : bool
        Whether to use GPU
    output_dir : str
        Output directory
    skip_scaling : bool
        If True, skips the StandardScaler step
    data : dict or None
        Full data dictionary (optional)
    save_full_model : bool
        Whether to save the model trained on full data
    save_train_model : bool
        Whether to save the model trained on training data only
    compute_shap : bool
        Whether to compute SHAP values for model interpretability
    
    Returns:
    --------
    dict
        Result dictionary
    """
    target_start = time.time()
    
    # Define file paths based on model saving options
    if save_train_model:
        train_model_filename = os.path.join(output_dir, 'models', f"{target_id}_train_model.pkl")
    else:
        train_model_filename = None
        
    if save_full_model:
        full_model_filename = os.path.join(output_dir, 'models', f"{target_id}_model.pkl")
    else:
        full_model_filename = None
    
    # The main model file for results will be either the full model or the train model
    if save_full_model:
        model_filename = full_model_filename
    else:
        model_filename = train_model_filename
    
    result_filename = os.path.join(output_dir, 'results', f"{target_id}_results.txt")
    
    # Create directory for predictions
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    # For classification, we might need to convert y to integer labels
    if not np.issubdtype(y_train.dtype, np.integer):
        # Get unique classes and map them to integers
        classes = np.unique(np.concatenate([y_train, y_test]))
        class_map = {val: idx for idx, val in enumerate(classes)}
        y_train_mapped = np.array([class_map[val] for val in y_train])
        y_test_mapped = np.array([class_map[val] for val in y_test])
        print(f"Mapped classes: {class_map}")
    else:
        class_map = None
        y_train_mapped = y_train
        y_test_mapped = y_test
    
    # Create a pipeline for X that scales and selects features
    pipeline_fs = create_feature_selection_pipeline(
        selected_k=selected_k, 
        methods=[method],
        verbose=verbose,
        gpu=gpu,
        task='classification',
        skip_scaling=skip_scaling
    )
    
    # Fit the pipeline on X_train
    pipeline_fs.fit(X_train, y_train_mapped, feature_selection__feature_names=predictor_names)

    # Transform both training and test sets
    X_train_selected = pipeline_fs.transform(X_train)
    X_test_selected = pipeline_fs.transform(X_test)
    
    # Save feature importance after fitting
    feature_importance_file = pipeline_fs.named_steps['feature_selection'].save_feature_importance(output_dir, target_id)
    
    # Get the indices of selected features
    selected_feature_indices = pipeline_fs.named_steps['feature_selection'].get_support(indices=True)
    
    # Get the names of selected features
    selected_feature_names = predictor_names[selected_feature_indices]
    
    # Concatenate the selected features from X with the covariates
    X_train_final = np.hstack([X_train_selected, covs_train]) if include_covariates else X_train_selected
    X_test_final = np.hstack([X_test_selected, covs_test]) if include_covariates else X_test_selected
    
    # Update categorical indices for the final feature matrix
    final_categorical_indices = categorical_indices.copy()
    if include_covariates:
        for cov_idx in cov_categorical_indices:
            final_categorical_indices.append(X_train_selected.shape[1] + cov_idx)
            
    # Initialize and train the classifier on the final training features
    device = "cuda:3" if gpu and torch.cuda.is_available() else "cpu"
    classifier = TabPFNClassifier(
        device=device,
        categorical_features_indices=final_categorical_indices,
        model_path = MODEL_CLS_PATH,
        n_estimators=8
    )
    classifier.fit(X_train_final, y_train_mapped)
    
    # Feature types list depends on whether covariates are included
    feature_types = ['predictor'] * len(selected_feature_names)
    feature_names = selected_feature_names
    
    if include_covariates:
        feature_types.extend(['covariate'] * len(covariate_names))
        feature_names = np.concatenate([selected_feature_names, covariate_names])
    
    # Predict on both training and test sets
    train_predictions = classifier.predict(X_train_final)
    test_predictions = classifier.predict(X_test_final)
    
    # Get probability outputs for both sets if available
    try:
        train_pred_proba = classifier.predict_proba(X_train_final)
        test_pred_proba = classifier.predict_proba(X_test_final)
        has_probas = True
    except Exception as e:
        print(f"Warning: predict_proba not available or failed: {e}")
        train_pred_proba = test_pred_proba = None
        has_probas = False
    
    # Map the predictions back to original classes if needed
    if class_map:
        reverse_map = {idx: val for val, idx in class_map.items()}
        train_pred_original = np.array([reverse_map[idx] for idx in train_predictions])
        test_pred_original = np.array([reverse_map[idx] for idx in test_predictions])
    else:
        train_pred_original = train_predictions
        test_pred_original = test_predictions
    
    # Save training and test set predictions with sample IDs
    train_pred_df = pd.DataFrame({
        'sample_id': sample_id_train,  # Add sample IDs
        'set': ['train'] * len(train_predictions),
        'true_value': y_train,
        'prediction': train_pred_original
    })
    
    test_pred_df = pd.DataFrame({
        'sample_id': sample_id_test,  # Add sample IDs
        'set': ['test'] * len(test_predictions),
        'true_value': y_test,
        'prediction': test_pred_original
    })
    
    # Add probability columns if available
    if has_probas:
        for i in range(train_pred_proba.shape[1]):
            class_name = reverse_map[i] if class_map else i
            train_pred_df[f'prob_class_{class_name}'] = train_pred_proba[:, i]
            test_pred_df[f'prob_class_{class_name}'] = test_pred_proba[:, i]
    
    # Combine the dataframes
    all_pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
    
    # Save predictions to file
    predictions_file = os.path.join(predictions_dir, f"{target_id}_training_predictions.csv.gz")
    all_pred_df.to_csv(predictions_file, index=False, compression='gzip')
    print(f"Saved training and test predictions to {predictions_file} (gzip compressed)")
    
    # Calculate classification metrics using test set
    accuracy = accuracy_score(y_test_mapped, test_predictions)
    
    # For multi-class, use 'weighted' average
    average_method = 'binary' if target_nlevels == 2 else 'weighted'
    
    try:
        precision = precision_score(y_test_mapped, test_predictions, average=average_method)
        recall = recall_score(y_test_mapped, test_predictions, average=average_method)
        f1 = f1_score(y_test_mapped, test_predictions, average=average_method)
    except Exception as e:
        print(f"Warning: Could not calculate some metrics: {e}")
        precision = recall = f1 = np.nan
    
    # Create a confusion matrix
    cm = confusion_matrix(y_test_mapped, test_predictions)
    print(f"Confusion matrix:\n{cm}")

    # Initialize shap_file variable
    shap_file = None
    
    # Compute SHAP values if requested
    if compute_shap:
        try:
            shap_file = compute_and_save_shap_values(
                classifier, 
                X_test_final, 
                feature_names, 
                output_dir, 
                target_id, 
                n_samples=min(50, X_test_final.shape[0]),
                is_classifier=True
            )
            print(f"Computed and saved SHAP values to {shap_file}")
        except Exception as e:
            print(f"Warning: Could not compute SHAP values: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping SHAP value computation as requested")
    
    # If requested, save the training model
    if save_train_model:        
        with open(train_model_filename, 'wb') as f:
            pickle.dump({
                'model': classifier, 
                'class_map': class_map,
                'target_id': target_id,
                'target_type': 'discrete',
                'include_covariates': include_covariates,
                'scale_features': not skip_scaling,
                'compute_shap': compute_shap,
                'framework_version': __version__,
                'feature_info': {
                    'feature_name': feature_names,
                    'feature_type': feature_types,
                    'categorical_indices': final_categorical_indices},
                'performance': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': cm.tolist()
                },
                'feature_importance_file': feature_importance_file,
                'predictions_file': predictions_file,
                'is_full_data_model': False,
                'is_train_data_model': True,
                'shap_values_file': shap_file
            }, f)
        print(f"Saved classification model (trained on TRAINING data) to {train_model_filename}")
    
    # Create a dictionary for this target's results
    result_dict = {
        "Target": target_name,
        "Target ID": target_id,
        "Type": "Classification",
        "Framework_Version": __version__,
        "Number of samples": len(y_train) + len(y_test),
        "Include Covariates": include_covariates,
        "Scale Features": not skip_scaling,
        "Compute SHAP": compute_shap,
        "Save Full Model": save_full_model,
        "Save Train Model": save_train_model,
        "Classes": target_nlevels,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Time (s)": time.time() - target_start,
        "Model file": model_filename,
        "Feature importance file": feature_importance_file,
        "Predictions file": predictions_file
    }
    
    # Add SHAP file to result dictionary if it was computed
    if shap_file:
        result_dict["SHAP values file"] = shap_file
    
    # Only retrain the model using ALL data if requested
    if save_full_model and (len(X_train) + len(X_test)) <= 10000:
        print("Retraining the model using all available data...")
        
        # Combine train and test data
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        covs_all = np.vstack([covs_train, covs_test]) if include_covariates else np.zeros((len(y_all), 0))
        sample_id_all = pd.concat([sample_id_train, sample_id_test]).reset_index(drop=True)  # Combine sample IDs

        # Map all labels if needed
        if class_map is not None:
            y_all_mapped = np.array([class_map[val] for val in y_all])
        else:
            y_all_mapped = y_all

        # Refit the feature selection pipeline on all data
        pipeline_fs.fit(X_all, y_all_mapped, feature_selection__feature_names=predictor_names)
        X_all_selected = pipeline_fs.transform(X_all)

        # Save feature importance for all-data model
        all_feature_importance_file = pipeline_fs.named_steps['feature_selection'].save_feature_importance(output_dir, f"{target_id}_full")

        # Verify that the same features are selected (they may be different when using all data)
        all_selected_feature_indices = pipeline_fs.named_steps['feature_selection'].get_support(indices=True)
        all_selected_feature_names = predictor_names[all_selected_feature_indices]

        print(f"Selected features using all data: {all_selected_feature_names}")

        # Prepare final feature matrix
        X_all_final = np.hstack([X_all_selected, covs_all]) if include_covariates else X_all_selected

        # Update categorical indices for the all-data feature matrix
        all_final_categorical_indices = categorical_indices.copy()
        if include_covariates:
            for cov_idx in cov_categorical_indices:
                all_final_categorical_indices.append(X_all_selected.shape[1] + cov_idx)

        # Initialize and train the final classifier on all data
        final_classifier = TabPFNClassifier(
            device=device,
            categorical_features_indices=all_final_categorical_indices,
            model_path = MODEL_CLS_PATH,
            n_estimators=8
        )
        final_classifier.fit(X_all_final, y_all_mapped)

        # Get predictions for all data
        all_predictions = final_classifier.predict(X_all_final)

        # Get probability outputs if available
        try:
            all_pred_proba = final_classifier.predict_proba(X_all_final)
            all_has_probas = True
        except Exception as e:
            print(f"Warning: predict_proba not available for all data: {e}")
            all_pred_proba = None
            all_has_probas = False

        # Map predictions back to original classes if needed
        if class_map:
            all_pred_original = np.array([reverse_map[idx] for idx in all_predictions])
        else:
            all_pred_original = all_predictions

        # Save all-data predictions with sample IDs
        all_data_pred_df = pd.DataFrame({
            'sample_id': sample_id_all,  # Add sample IDs
            'set': ['all'] * len(all_predictions),
            'true_value': y_all,
            'prediction': all_pred_original
        })

        # Add probability columns if available
        if all_has_probas:
            for i in range(all_pred_proba.shape[1]):
                class_name = reverse_map[i] if class_map else i
                all_data_pred_df[f'prob_class_{class_name}'] = all_pred_proba[:, i]

        all_predictions_file = os.path.join(predictions_dir, f"{target_id}_full_predictions.csv.gz")
        all_data_pred_df.to_csv(all_predictions_file, index=False, compression='gzip')
        print(f"Saved full data predictions to {all_predictions_file} (gzip compressed)")

        # Initialize all_shap_file variable
        all_shap_file = None
        
        # Compute SHAP values for full model if requested
        if compute_shap:
            try:
                all_shap_file = compute_and_save_shap_values(
                    final_classifier, 
                    X_all_final, 
                    all_selected_feature_names if not include_covariates else 
                    np.concatenate([all_selected_feature_names, covariate_names]), 
                    output_dir, 
                    f"{target_id}_full", 
                    n_samples=min(50, X_all_final.shape[0]),
                    is_classifier=True
                )
                print(f"Computed and saved SHAP values for full model to {all_shap_file}")
            except Exception as e:
                print(f"Warning: Could not compute SHAP values for full model: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping SHAP value computation for full model as requested")

        # Update feature information
        all_feature_types = ['predictor'] * len(all_selected_feature_names)
        all_feature_names = all_selected_feature_names

        if include_covariates:
            all_feature_types.extend(['covariate'] * len(covariate_names))
            all_feature_names = np.concatenate([all_selected_feature_names, covariate_names])
        
        # Save the ALL-DATA trained model
        with open(full_model_filename, 'wb') as f:
            pickle.dump({
                'model': final_classifier, 
                'class_map': class_map,
                'target_id': target_id,
                'target_type': 'discrete',
                'include_covariates': include_covariates,
                'scale_features': not skip_scaling,
                'compute_shap': compute_shap,
                'framework_version': __version__,
                'feature_info': {
                    'feature_name': all_feature_names,
                    'feature_type': all_feature_types,
                    'categorical_indices': all_final_categorical_indices},
                'performance': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': cm.tolist()
                },
                'feature_importance_file': feature_importance_file,
                'all_feature_importance_file': all_feature_importance_file,
                'predictions_file': predictions_file,
                'all_predictions_file': all_predictions_file,
                'is_full_data_model': True,
                'is_train_data_model': False,
                'shap_values_file': shap_file,
                'all_shap_values_file': all_shap_file
            }, f)
        print(f"Saved classification model (trained on ALL data) to {full_model_filename}")
        
        # Update the result dictionary with full-data model info
        result_dict.update({
            "All Feature importance file": all_feature_importance_file,
            "All Predictions file": all_predictions_file
        })
        
        # Add SHAP file to result dictionary if it was computed
        if all_shap_file:
            result_dict["All SHAP values file"] = all_shap_file
    
    # Save the result_dict to an individual text file
    with open(result_filename, 'w') as f:
        # Format the output as pretty JSON
        json.dump(result_dict, f, indent=4)
    print(f"Saved individual result to {result_filename}")

    return result_dict