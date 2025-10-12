"""
Module for computing SHAP values from saved models
"""
import os
import numpy as np
import pandas as pd
import time
import argparse
import gc
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit

from prediction import load_model
from data_loader import load_data, get_samples_by_group
from model_training import compute_and_save_shap_values


def clear_gpu_memory():
    """Clear GPU memory if CUDA is available"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True
    except Exception:
        pass
    return False


def get_test_data_for_target(data, model_data, target_name):
    """
    Recreate the test data for the specified target using the same split strategy as during training
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the loaded data
    model_data : dict
        Dictionary containing the model metadata
    target_name : str
        Name of the target variable
        
    Returns:
    --------
    tuple
        (X_test, covs_test, sample_id_test)
    """
    # Get valid samples for this target (non-NA values)
    valid_idx = data['Y_df'][target_name].notna()
    X_valid = data['X'][valid_idx, :]
    y = data['Y_df'][target_name][valid_idx].values
    covs_valid = data['Covs'][valid_idx, :] if model_data.get('include_covariates', False) else np.zeros((sum(valid_idx), 0))
    sample_id_valid = data['sample_id'][valid_idx].reset_index(drop=True)
    
    # Extract person_id from sample_id for person-level validation
    sample_id_valid = sample_id_valid.astype(str)
    person_ids = sample_id_valid.str.split('_', expand=True)[0]
    
    # Check if we used sample group-based split
    if ('Test Sample Groups' in model_data and 
        model_data['Test Sample Groups'] and 
        model_data['Test Sample Groups'][0] != 'random_split'):
        
        print(f"Using sample group-based split with groups: {', '.join(model_data['Test Sample Groups'])}")
        
        # Get mask for samples in the specified test groups
        test_mask = get_samples_by_group(
            sample_id_valid, 
            data['sample_annot_df'], 
            model_data['Test Sample Groups']
        )
        
        # If no test samples were found, fall back to GroupShuffleSplit
        if test_mask is None or sum(test_mask) == 0:
            print("Warning: No test samples found in the specified groups. Using person ID-based split.")
            gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
            _, test_idx = next(gss.split(X_valid, y, groups=person_ids))
        else:
            # Use the test_mask to determine test indices
            test_idx = np.where(test_mask)[0]
            print(f"Sample group split: {len(test_idx)} test samples")
    else:
        # Use GroupShuffleSplit with the same random_state
        print("Using person ID-based split")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        _, test_idx = next(gss.split(X_valid, y, groups=person_ids))
        print(f"Person ID-based split: {len(test_idx)} test samples")
    
    # Get test data
    X_test = X_valid[test_idx]
    covs_test = covs_valid[test_idx] if model_data.get('include_covariates', False) else np.zeros((len(test_idx), 0))
    sample_id_test = sample_id_valid.iloc[test_idx]
    
    return X_test, covs_test, sample_id_test


def prepare_features_for_shap(model_data, X_test, covs_test, data):
    """
    Prepare features for SHAP computation using the same features as the model
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing the model metadata
    X_test : numpy.ndarray
        Test feature matrix
    covs_test : numpy.ndarray
        Test covariate matrix
    data : dict
        Dictionary containing the loaded data
        
    Returns:
    --------
    numpy.ndarray
        Prepared feature matrix for SHAP computation
    """
    # Get feature information from the model
    feature_info = model_data['feature_info']
    feature_names = feature_info['feature_name']
    feature_types = feature_info['feature_type']
    
    # Separate predictor and covariate feature names
    predictor_names = [name for name, type_ in zip(feature_names, feature_types) 
                     if type_ == 'predictor']
    
    # Get original predictor names from data
    original_predictor_names = data['predictor_names']
    
    # Reorder the test data to match the model's predictor order
    X_test_reordered = np.zeros((X_test.shape[0], len(predictor_names)))
    for i, feature_name in enumerate(predictor_names):
        if feature_name in original_predictor_names:
            feature_idx = np.where(original_predictor_names == feature_name)[0][0]
            X_test_reordered[:, i] = X_test[:, feature_idx]
        else:
            print(f"Warning: Feature '{feature_name}' not found in test data. Using zeros.")
    
    # Apply scaling if the model was trained with scaling
    if model_data.get('scale_features', True):
        print("Applying StandardScaler to features (model was trained with scaled data)")
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test_reordered)
    else:
        print("Skipping feature scaling (model was trained with unscaled data)")
        X_test_scaled = X_test_reordered
    
    # If the model uses covariates, include them
    if model_data.get('include_covariates', False) and covs_test is not None and covs_test.shape[1] > 0:
        covariate_names = [name for name, type_ in zip(feature_names, feature_types) 
                          if type_ == 'covariate']
        
        # Get original covariate names from data
        original_covariate_names = data['covariate_names']
        
        # Reorder covariates to match the model's order
        covs_test_reordered = np.zeros((covs_test.shape[0], len(covariate_names)))
        for i, feature_name in enumerate(covariate_names):
            if feature_name in original_covariate_names:
                feature_idx = np.where(original_covariate_names == feature_name)[0][0]
                covs_test_reordered[:, i] = covs_test[:, feature_idx]
            else:
                print(f"Warning: Covariate '{feature_name}' not found in test data. Using zeros.")
        
        # Combine predictors and covariates
        X_final = np.hstack([X_test_scaled, covs_test_reordered])
        print(f"Final feature matrix includes {len(predictor_names)} predictors and {len(covariate_names)} covariates")
    else:
        X_final = X_test_scaled
        print(f"Final feature matrix includes {len(predictor_names)} predictors and no covariates")
    
    return X_final


def sample_data_for_shap(X, sample_ids=None, n_samples=50, random_state=42):
    """
    Sample data for SHAP computation to manage computation time
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    sample_ids : pandas.Series, optional
        Sample IDs corresponding to X
    n_samples : int, default=50
        Maximum number of samples to use
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (X_sampled, sample_ids_sampled) if sample_ids is provided, otherwise X_sampled
    """
    if X.shape[0] <= n_samples:
        # No sampling needed
        if sample_ids is not None:
            return X, sample_ids
        return X
    
    print(f"Sampling {n_samples} out of {X.shape[0]} samples for SHAP computation")
    
    # Perform sampling
    if sample_ids is not None:
        # Sample from indices and return corresponding sample_ids
        indices = np.arange(len(X))
        sampled_indices = np.sort(np.random.RandomState(random_state).choice(
            indices, size=n_samples, replace=False))
        X_sampled = X[sampled_indices]
        sample_ids_sampled = sample_ids.iloc[sampled_indices].reset_index(drop=True)
        return X_sampled, sample_ids_sampled
    else:
        # Sample directly from X
        np.random.seed(random_state)
        sampled_indices = np.random.choice(X.shape[0], n_samples, replace=False)
        return X[sampled_indices]


def prepare_test_data_for_shap(model_data, data_dir, n_samples=50):
    """
    Prepare test data for SHAP value computation based on a saved model
    
    Parameters:
    -----------
    model_data : dict
        Loaded model data dictionary
    data_dir : str
        Directory containing the original data files
    n_samples : int, default=50
        Maximum number of samples to use for SHAP computation
        
    Returns:
    --------
    tuple
        (X_prepared, feature_names, sample_ids)
    """
    # Get target information
    target_id = model_data['target_id']
    target_name = None
    
    # Find target name from target_id
    data = load_data(
        data_dir=data_dir,
        include_covariates=model_data.get('include_covariates', False),
        predictor_group=model_data.get('Predictor Groups', None)
    )
    
    # Find target name from target_id in Y_annot_df
    target_row = data['Y_annot_df'][data['Y_annot_df']['target_id'] == target_id]
    if not target_row.empty:
        target_name = target_row['name'].values[0]
    else:
        raise ValueError(f"Could not find target name for target_id {target_id}")
    
    print(f"Preparing test data for target: {target_name} (ID: {target_id})")
    
    # Get test data
    X_test, covs_test, sample_id_test = get_test_data_for_target(data, model_data, target_name)
    
    # Prepare features
    X_final = prepare_features_for_shap(model_data, X_test, covs_test, data)
    
    # Sample data if needed
    if X_final.shape[0] > n_samples:
        X_final, sample_id_test = sample_data_for_shap(X_final, sample_id_test, n_samples)
    
    # Get feature names from model
    feature_names = model_data['feature_info']['feature_name']
    
    return X_final, feature_names, sample_id_test


def compute_shap_for_saved_model(model_path, data_dir, output_dir=None, n_samples=50, force=False):
    """
    Compute SHAP values for a saved model using original test data
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    data_dir : str
        Directory containing the original data files
    output_dir : str, optional
        Directory to save SHAP values (defaults to same directory as model)
    n_samples : int, default=50
        Maximum number of samples to use for SHAP computation
    force : bool, default=False
        Whether to force recomputation even if SHAP values already exist
        
    Returns:
    --------
    str
        Path to the saved SHAP values file
    """
    start_time = time.time()
    
    print(f"Loading model from {model_path}")
    model_data = load_model(model_path)
    
    if model_data is None:
        raise ValueError(f"Could not load model from {model_path}")
    
    # Extract model information
    model = model_data['model']
    target_id = model_data['target_id']
    target_type = model_data['target_type']
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(model_path))
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Expected SHAP file path based on target ID
    shap_dir = os.path.join(output_dir, 'shap_values')
    expected_shap_file = os.path.join(shap_dir, f"{target_id}_shap_values.csv.gz")
    
    # Check if SHAP values already exist - either from training or previous computations
    if not force:
        # Check if SHAP values were computed during training
        if 'shap_values_file' in model_data and model_data['shap_values_file']:
            existing_shap_file = model_data['shap_values_file']
            if os.path.exists(existing_shap_file):
                print(f"SHAP values already exist (from training): {existing_shap_file}")
                print(f"Skipping computation. Use --force flag to recompute if needed.")
                return existing_shap_file
        
        # Check if SHAP values were computed separately after training
        if os.path.exists(expected_shap_file):
            print(f"SHAP values already exist (from previous computation): {expected_shap_file}")
            print(f"Skipping computation. Use --force flag to recompute if needed.")
            return expected_shap_file
    
    # If we get here, we need to compute SHAP values
    print(f"Preparing test data from {data_dir}")
    X_final, feature_names, sample_ids = prepare_test_data_for_shap(
        model_data, data_dir, n_samples
    )
    
    print(f"Prepared {X_final.shape[0]} samples for SHAP computation")
    
    # Compute SHAP values
    print(f"Computing SHAP values...")
    
    shap_file = compute_and_save_shap_values(
        model=model,
        X_test_sample=X_final,
        feature_names=feature_names,
        output_dir=output_dir,
        target_id=target_id,
        sample_ids=sample_ids,  # Pass the sample IDs
        n_samples=X_final.shape[0],  # Using all prepared samples
        is_classifier=(target_type == 'discrete')
    )
    
    if shap_file is None:
        print(f"Warning: Failed to compute SHAP values for {target_id}")
    else:
        print(f"Successfully computed SHAP values: {shap_file}")
    
    # Clear GPU memory if available
    if clear_gpu_memory():
        print("Cleared GPU memory after SHAP computation")
    
    # Cleanup
    gc.collect()
    
    end_time = time.time()
    print(f"SHAP computation completed in {end_time - start_time:.2f} seconds")
    
    return shap_file


def process_multiple_models(models_dir, data_dir, output_dir=None, n_samples=50, target_ids=None, force=False):
    """
    Process multiple models in the specified directory
    
    Parameters:
    -----------
    models_dir : str
        Directory containing model files
    data_dir : str
        Directory containing the original data files
    output_dir : str, optional
        Directory to save SHAP values
    n_samples : int, default=50
        Maximum number of samples to use for SHAP computation
    target_ids : list, optional
        List of specific target IDs to process
        
    Returns:
    --------
    dict
        Dictionary containing results for each processed model
    """
    import glob
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(models_dir))
    
    # Find model files
    if target_ids:
        # If specific target IDs are provided, only look for those models
        model_files = []
        for target_id in target_ids:
            model_path = os.path.join(models_dir, f"{target_id}_model.pkl")
            if os.path.exists(model_path):
                model_files.append(model_path)
            else:
                print(f"Warning: Model file for target_id '{target_id}' not found at {model_path}")
    else:
        # Otherwise, find all model files
        model_files = glob.glob(os.path.join(models_dir, '*_model.pkl'))
    
    print(f"Found {len(model_files)} model files to process")
    
    # Process each model
    results = {}
    for i, model_file in enumerate(model_files):
        target_id = os.path.basename(model_file).split('_model.pkl')[0]
        print(f"\nProcessing model {i+1}/{len(model_files)}: {target_id}")
        
        try:
            shap_file = compute_shap_for_saved_model(
                model_file, data_dir, output_dir, n_samples, force
            )
            results[target_id] = {
                'status': 'success' if shap_file else 'failed',
                'shap_file': shap_file
            }
        except Exception as e:
            print(f"Error processing model {target_id}: {e}")
            import traceback
            traceback.print_exc()
            results[target_id] = {
                'status': 'error',
                'error_message': str(e)
            }
    
    # Summarize results
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    failed_count = sum(1 for r in results.values() if r['status'] == 'failed')
    error_count = sum(1 for r in results.values() if r['status'] == 'error')
    
    print(f"\nProcessed {len(results)} models:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Error: {error_count}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute SHAP values for saved models')
    parser.add_argument('--model_path', type=str, 
                        help='Path to a specific model file')
    parser.add_argument('--models_dir', type=str, 
                        help='Directory containing model files')
    parser.add_argument('--data_dir', type=str, default='../data', 
                        help='Directory containing the original data files')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save SHAP values (defaults to model directory)')
    parser.add_argument('--n_samples', type=int, default=50, 
                        help='Maximum number of samples to use for SHAP computation')
    parser.add_argument('--target_ids', type=str, nargs='+', 
                        help='Specific target IDs to process')
    parser.add_argument('--force', action='store_true',
                        help='Force recomputation of SHAP values even if they already exist')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model_path and not args.models_dir:
        parser.error("At least one of --model_path or --models_dir is required")
    
    if args.model_path and args.models_dir:
        print("Both --model_path and --models_dir provided. Will process the specific model only.")
    
    if args.model_path:
        # Process a single model
        print(f"Processing single model: {args.model_path}")
        shap_file = compute_shap_for_saved_model(
            args.model_path, 
            args.data_dir,
            args.output_dir,
            args.n_samples,
            args.force
        )
        
        if shap_file:
            print(f"SHAP values saved to: {shap_file}")
        else:
            print("Failed to compute SHAP values")
    else:
        # Process multiple models
        print(f"Processing multiple models from: {args.models_dir}")
        results = process_multiple_models(
            args.models_dir,
            args.data_dir,
            args.output_dir,
            args.n_samples,
            args.target_ids,
            args.force
        )
        
        # Save results summary
        if args.output_dir:
            summary_path = os.path.join(args.output_dir, 'shap_computation_summary.csv')
            summary_data = []
            
            for target_id, result in results.items():
                summary_data.append({
                    'target_id': target_id,
                    'status': result['status'],
                    'shap_file': result.get('shap_file', ''),
                    'error_message': result.get('error_message', '')
                })
            
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            print(f"Saved results summary to: {summary_path}")


if __name__ == "__main__":
    main()