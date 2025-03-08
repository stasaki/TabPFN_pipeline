"""
Module for training and evaluating models
"""
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
from sklearn.model_selection import GroupShuffleSplit
from tabpfn import TabPFNRegressor, TabPFNClassifier
from version import __version__

from feature_selection import create_feature_selection_pipeline
from data_loader import check_existing_results, load_existing_result

def process_target(data, target_name, include_covariates, selected_k, method="CatBoost", verbose=1, gpu=True, output_dir='.', skip_scaling=False):
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
              (f", using pre-scaled data" if skip_scaling else ", scaling data"))
        
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
        person_ids = sample_id_valid.str.split('_', expand=True)[0]  # Get the first part before underscore
        
        # Number of samples for this target
        num_samples = len(y)
        print(f"Number of valid samples: {num_samples}")
        
        if num_samples < 50:
            print(f"Warning: Only {num_samples} samples for target {target_name}. Skipping due to insufficient data.")
            return None
        
        # Use GroupShuffleSplit to ensure the same person isn't in both train and test
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        
        # Generate indices for train and test sets
        train_idx, test_idx = next(gss.split(X_valid, y, groups=person_ids))
        
        # Split the data using these indices
        X_train, X_test = X_valid[train_idx], X_valid[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        covs_train, covs_test = covs_valid[train_idx], covs_valid[test_idx] if include_covariates else (np.zeros((len(train_idx), 0)), np.zeros((len(test_idx), 0)))
        
        # Verify the split respects person_ids
        train_people = set(person_ids.iloc[train_idx])
        test_people = set(person_ids.iloc[test_idx])
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
                selected_k, data['predictor_names'], data['covariate_names'],
                cov_categorical_indices, categorical_indices,
                target_name, target_id, include_covariates,
                method, verbose, gpu, output_dir, skip_scaling
            )
            
        elif target_type == 'discrete':
            # CLASSIFICATION WORKFLOW
            result_dict = train_classification_model(
                X_train, X_test, y_train, y_test,
                covs_train, covs_test,
                selected_k, data['predictor_names'], data['covariate_names'],
                cov_categorical_indices, categorical_indices,
                target_name, target_id, target_nlevels, include_covariates,
                method, verbose, gpu, output_dir, skip_scaling
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
        
# Add functions to train classification and regression models


def train_regression_model(X_train, X_test, y_train, y_test, 
                           covs_train, covs_test,
                           selected_k, predictor_names, covariate_names,
                           cov_categorical_indices, categorical_indices,
                           target_name, target_id, include_covariates,
                           method="CatBoost", verbose=1, gpu=True, output_dir='.', skip_scaling=False):
    """
    Train a regression model
    
    Parameters remain unchanged
    
    Returns:
    --------
    dict
        Result dictionary
    """
    target_start = time.time()
    model_filename = os.path.join(output_dir, 'models', f"{target_id}_model.pkl")
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
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    regressor = TabPFNRegressor(
        device=device,
        categorical_features_indices=final_categorical_indices,
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
    
    # Save training and test set predictions
    train_ids = np.arange(len(train_predictions))  # Using indices as placeholders
    test_ids = np.arange(len(test_predictions))    # Replace with actual IDs if available
    
    train_pred_df = pd.DataFrame({
        'set': ['train'] * len(train_predictions),
        'true_value': y_train,
        'prediction': train_predictions
    })
    
    test_pred_df = pd.DataFrame({
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
    
    # Create a dictionary for this target's results
    result_dict = {
        "Target": target_name,
        "Target ID": target_id,
        "Type": "Regression",
        "Framework_Version": __version__,
        "Number of samples": len(y_train) + len(y_test),
        "Include Covariates": include_covariates,
        "Scale Features": not skip_scaling,
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Pearson": pearson_r,
        "Time (s)": time.time() - target_start,
        "Model file": model_filename,
        "Feature importance file": feature_importance_file,
        "Predictions file": predictions_file
    }
    
    # Now retrain the model using ALL data after evaluation
    print("Retraining the regression model using all available data...")
    
    # Combine train and test data
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    covs_all = np.vstack([covs_train, covs_test]) if include_covariates else np.zeros((len(y_all), 0))
    
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
        n_estimators=8
    )
    final_regressor.fit(X_all_final, y_all_scaled)
    
    # Get predictions from the final model on all data
    all_predictions_scaled = final_regressor.predict(X_all_final)
    all_predictions = y_all_scaler.inverse_transform(all_predictions_scaled.reshape(-1, 1)).ravel()
    
    # Save all-data predictions
    all_data_pred_df = pd.DataFrame({
        'set': ['all'] * len(all_predictions),
        'true_value': y_all,
        'prediction': all_predictions
    })
    
    all_predictions_file = os.path.join(predictions_dir, f"{target_id}_full_predictions.csv.gz")
    all_data_pred_df.to_csv(all_predictions_file, index=False, compression='gzip')
    print(f"Saved full data predictions to {all_predictions_file} (gzip compressed)")
    
    # Update feature information
    all_feature_types = ['predictor'] * len(all_selected_feature_names)
    all_feature_names = all_selected_feature_names
    
    if include_covariates:
        all_feature_types.extend(['covariate'] * len(covariate_names))
        all_feature_names = np.concatenate([all_selected_feature_names, covariate_names])
    
    # Save the ALL-DATA trained model
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': final_regressor, 
            'y_scaler': y_all_scaler,
            'target_id': target_id,
            'target_type': 'continuous',
            'include_covariates': include_covariates,
            'scale_features': not skip_scaling,
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
            'is_full_data_model': True
        }, f)
    print(f"Saved regression model (trained on ALL data) to {model_filename}")
    
    # Save the result_dict to an individual text file
    with open(result_filename, 'w') as f:
        # Format the output as pretty JSON
        json.dump(result_dict, f, indent=4)
    print(f"Saved individual result to {result_filename}")
    
    return result_dict

def train_classification_model(X_train, X_test, y_train, y_test, 
                              covs_train, covs_test,
                              selected_k, predictor_names, covariate_names,
                              cov_categorical_indices, categorical_indices,
                              target_name, target_id, target_nlevels, include_covariates,
                              method="CatBoost", verbose=1, gpu=True, output_dir='.', skip_scaling=False):
    """
    Train a classification model
    
    Parameters remain unchanged
    
    Returns:
    --------
    dict
        Result dictionary
    """
    target_start = time.time()
    model_filename = os.path.join(output_dir, 'models', f"{target_id}_model.pkl")
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
    device = "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    classifier = TabPFNClassifier(
        device=device,
        categorical_features_indices=final_categorical_indices,
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
    
    # Save training and test set predictions
    train_pred_df = pd.DataFrame({
        'set': ['train'] * len(train_predictions),
        'true_value': y_train,
        'prediction': train_pred_original
    })
    
    test_pred_df = pd.DataFrame({
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
    
    # Create a dictionary for this target's results
    result_dict = {
        "Target": target_name,
        "Target ID": target_id,
        "Type": "Classification",
        "Framework_Version": __version__,
        "Number of samples": len(y_train) + len(y_test),
        "Include Covariates": include_covariates,
        "Scale Features": not skip_scaling,
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
    
    # Now retrain the model using ALL data after evaluation
    print("Retraining the model using all available data...")
    
    # Combine train and test data
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    covs_all = np.vstack([covs_train, covs_test]) if include_covariates else np.zeros((len(y_all), 0))
    
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
    
    # Save all-data predictions
    all_data_pred_df = pd.DataFrame({
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

    # Update feature information
    all_feature_types = ['predictor'] * len(all_selected_feature_names)
    all_feature_names = all_selected_feature_names
    
    if include_covariates:
        all_feature_types.extend(['covariate'] * len(covariate_names))
        all_feature_names = np.concatenate([all_selected_feature_names, covariate_names])
    
    # Save the ALL-DATA trained model
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': final_classifier, 
            'class_map': class_map,
            'target_id': target_id,
            'target_type': 'discrete',
            'include_covariates': include_covariates,
            'scale_features': not skip_scaling,
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
            'is_full_data_model': True
        }, f)
    print(f"Saved classification model (trained on ALL data) to {model_filename}")
    
    # Save the result_dict to an individual text file
    with open(result_filename, 'w') as f:
        # Format the output as pretty JSON
        json.dump(result_dict, f, indent=4)
    print(f"Saved individual result to {result_filename}")
    
    return result_dict