"""
Module for making predictions using trained models
"""
import numpy as np
import pandas as pd
import pickle
import os
import torch
from sklearn.preprocessing import StandardScaler
import glob
import warnings
from data_loader import load_data

def load_model(model_path):
    """
    Load a trained model from a pickle file
    
    Parameters:
    -----------
    model_path : str
        Path to the pickled model file
        
    Returns:
    --------
    dict
        Dictionary containing the model and associated metadata
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

def predict_with_model(model_data, X_data, Covs_data=None, predictor_names=None, covariate_names=None, sample_id=None):
    """
    Make predictions using a loaded model
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing the model and associated metadata
    X_data : numpy.ndarray
        Feature matrix for prediction
    Covs_data : numpy.ndarray, optional
        Covariate matrix for prediction
    predictor_names : numpy.ndarray, optional
        Names of predictors/features
    covariate_names : numpy.ndarray, optional
        Names of covariates
    sample_id : numpy.ndarray, optional
        Sample IDs
        
    Returns:
    --------
    tuple
        Predictions and additional metadata (probabilities for classification)
    """
    if model_data is None:
        print("No model data provided")
        return None
    
    # Extract model components
    model = model_data['model']
    target_id = model_data['target_id']
    target_type = model_data['target_type']
    feature_info = model_data['feature_info']
    
    # Check if model was trained with covariates and scaling
    # Default to False for backwards compatibility with older models
    model_uses_covariates = model_data.get('include_covariates', False)
    model_uses_scaling = model_data.get('scale_features', True)  # Default to True for backwards compatibility
    
    print(f"Model settings - uses covariates: {model_uses_covariates}, scales features: {model_uses_scaling}")
    
    # Get feature names and types from the model
    model_feature_names = feature_info['feature_name']
    model_feature_types = feature_info['feature_type']
    model_categorical_indices = feature_info.get('categorical_indices', [])
    
    # Separate predictor and covariate features using feature_type
    model_predictor_names = [name for name, type_ in zip(model_feature_names, model_feature_types) if type_ == 'predictor']
    model_covariate_names = [name for name, type_ in zip(model_feature_names, model_feature_types) if type_ == 'covariate']
    
    print(f"Model uses {len(model_predictor_names)} predictors and {len(model_covariate_names)} covariates")
    
    # Reorder the X data to match the model's predictor feature order
    X_reordered = np.zeros((X_data.shape[0], len(model_predictor_names)))
    for i, feature_name in enumerate(model_predictor_names):
        if predictor_names is not None and feature_name in predictor_names:
            feature_idx = np.where(predictor_names == feature_name)[0][0]
            X_reordered[:, i] = X_data[:, feature_idx]
        else:
            print(f"Warning: Feature '{feature_name}' not found in new data. Using zeros.")
    
    # Scale X data if the model was trained with scaling
    if model_uses_scaling:
        print("Applying StandardScaler to features (model was trained with scaled data)")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reordered)
    else:
        print("Skipping feature scaling (model was trained with unscaled data)")
        X_scaled = X_reordered
    
    # Handle covariates if the model uses them
    if model_uses_covariates and len(model_covariate_names) > 0:
        if Covs_data is None or Covs_data.shape[1] == 0:
            print("Warning: Model requires covariates but none provided. Using zeros.")
            Covs_reordered = np.zeros((X_data.shape[0], len(model_covariate_names)))
        else:
            # Reorder the covariates to match the model's covariate order
            Covs_reordered = np.zeros((Covs_data.shape[0], len(model_covariate_names)))
            for i, feature_name in enumerate(model_covariate_names):
                if covariate_names is not None and feature_name in covariate_names:
                    feature_idx = np.where(covariate_names == feature_name)[0][0]
                    Covs_reordered[:, i] = Covs_data[:, feature_idx]
                else:
                    print(f"Warning: Covariate '{feature_name}' not found in new data. Using zeros.")
        
        # Combine X and Covs for prediction
        X_final = np.hstack([X_scaled, Covs_reordered])
    else:
        # Model doesn't use covariates
        X_final = X_scaled
    
    # Set device for GPU acceleration if available
    if hasattr(model, 'device'):
        original_device = model.device
        if torch.cuda.is_available():
            try:
                model.device = "cuda:0"
                print(f"Changed device from {original_device} to {model.device}")
                
                # Force model to the correct device if it has a to() method
                if hasattr(model, 'to'):
                    model.to(model.device)
            except Exception as e:
                print(f"Could not set CUDA device: {e}")
                model.device = "cpu"
        else:
            model.device = "cpu"
    
    # Make predictions based on target type
    if target_type == 'continuous':
        # For regression models
        y_scaler = model_data['y_scaler']
        predictions_scaled = model.predict(X_final)
        predictions = y_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
        return predictions, None  # No probabilities for regression
    
    elif target_type == 'discrete':
        # For classification models
        predictions_idx = model.predict(X_final)
        
        # Convert indices back to original class labels if we have a class map
        if 'class_map' in model_data and model_data['class_map'] is not None:
            class_map = model_data['class_map']
            reverse_map = {idx: label for label, idx in class_map.items()}
            predictions = np.array([reverse_map[idx] for idx in predictions_idx])
        else:
            predictions = predictions_idx
            
        # Get prediction probabilities if possible
        try:
            pred_proba = model.predict_proba(X_final)
            return predictions, pred_proba
        except Exception as e:
            print(f"Warning: Could not get prediction probabilities: {e}")
            return predictions, None
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_prediction_exists(target_id, output_dir):
    """
    Check if prediction files already exist for a given target_id
    
    Parameters:
    -----------
    target_id : str
        Target ID
    output_dir : str
        Directory containing prediction files
        
    Returns:
    --------
    bool
        True if prediction files exist, False otherwise
    """
    prediction_file = os.path.join(output_dir, f"{target_id}_predictions.csv.gz")
    
    # Check if the prediction file exists
    exists = os.path.exists(prediction_file)
    
    if exists:
        try:
            # Try to read the file to ensure it's valid
            df = pd.read_csv(prediction_file)
            
            # Check if the file has the expected structure
            if 'sample_id' in df.columns and 'prediction' in df.columns and len(df) > 0:
                return True
            else:
                print(f"Warning: Existing prediction file for {target_id} has invalid structure. Will regenerate.")
                return False
        except Exception as e:
            print(f"Warning: Existing prediction file for {target_id} is corrupt or invalid: {e}. Will regenerate.")
            return False
    
    return False

def make_predictions(data_dir, models_dir, output_dir=None, force_covariates_mode=None, force_scaling_mode=None, skip_existing=True, target_ids=None):
    """
    Make predictions for models in the models directory
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
    models_dir : str
        Directory containing the model files
    output_dir : str, optional
        Directory to save the predictions (default is './predictions')
    force_covariates_mode : bool, optional
        If specified, force all predictions to include or exclude covariates
        regardless of model requirements. Useful for testing. 
        If None (default), each model will use covariates based on its own requirements.
    force_scaling_mode : bool, optional
        If specified, force all predictions to apply or skip feature scaling
        regardless of model requirements. Useful for testing.
        If None (default), each model will use scaling based on its own requirements.
    skip_existing : bool, optional
        If True, skip predictions that already exist in the output directory
    target_ids : list, optional
        List of specific target IDs to process. If None, all models will be processed.
        
    Returns:
    --------
    dict
        Dictionary containing all predictions
    """
    if output_dir is None:
        output_dir = './predictions'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # First, check if we need covariates for any model
    # This determines whether we need to load covariates data at all
    covariates_needed = False
    if force_covariates_mode is True:
        covariates_needed = True
    elif force_covariates_mode is None:
        # Check if any model needs covariates
        model_files = glob.glob(os.path.join(models_dir, '*_model.pkl'))
        for model_file in model_files:
            try:
                model_data = load_model(model_file)
                if model_data and model_data.get('include_covariates', False):
                    covariates_needed = True
                    break
            except Exception as e:
                print(f"Error checking covariate requirements in {model_file}: {e}")
    
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
        print(f"Found {len(model_files)} of {len(target_ids)} requested model files")
    else:
        # Otherwise, find all model files
        model_files = glob.glob(os.path.join(models_dir, '*_model.pkl'))
        print(f"Found {len(model_files)} model files")
    
    # Check for existing predictions if skip_existing is True
    if skip_existing:
        models_to_process = []
        skipped_models = []
        
        for model_file in model_files:
            target_id = os.path.basename(model_file).split('_model.pkl')[0]
            if check_prediction_exists(target_id, output_dir):
                skipped_models.append(target_id)
            else:
                models_to_process.append(model_file)
        
        if skipped_models:
            print(f"Skipping {len(skipped_models)} models with existing predictions: {', '.join(skipped_models)}")
        
        if not models_to_process:
            print("All predictions already exist. Nothing to do.")
            return
        
        model_files = models_to_process
        print(f"Will process {len(model_files)} models")
    
    # Load data with covariates if needed (only if we have models to process)
    if model_files:
        print(f"Loading data with include_covariates={covariates_needed}")
        data = load_data(data_dir, covariates_needed)
    else:
        return
    
    # Process each model and store results
    all_target_ids = []
    processed_count = 0
    
    for model_file in model_files:
        try:
            # Extract target ID from filename
            target_id = os.path.basename(model_file).split('_model.pkl')[0]
            all_target_ids.append(target_id)
            
            # Load the model
            model_data = load_model(model_file)
            if model_data is None:
                print(f"Skipping {target_id}: Could not load model")
                continue
                
            # Check if model requires covariates and scaling
            model_uses_covariates = model_data.get('include_covariates', False)
            model_uses_scaling = model_data.get('scale_features', True)  # Default to True for backwards compatibility
            
            # Determine whether to use covariates based on model requirements and force_covariates_mode
            use_covariates = model_uses_covariates
            if force_covariates_mode is not None:
                if force_covariates_mode and not model_uses_covariates:
                    print(f"Warning: Model {target_id} was trained without covariates but forced to use them")
                    # Still proceed, but model might ignore them
                use_covariates = force_covariates_mode
            
            # Determine whether to apply scaling based on model requirements and force_scaling_mode
            use_scaling = model_uses_scaling
            if force_scaling_mode is not None:
                if force_scaling_mode != model_uses_scaling:
                    print(f"Warning: Model {target_id} was trained with scale_features={model_uses_scaling} "
                          f"but prediction is using scale_features={force_scaling_mode}")
                use_scaling = force_scaling_mode
                
            print(f"Model {target_id} - uses_covariates={model_uses_covariates}, prediction uses_covariates={use_covariates}, "
                  f"uses_scaling={model_uses_scaling}, prediction uses_scaling={use_scaling}")
            
            # Make predictions
            predictions = predict_with_model(
                model_data, 
                data['X'], 
                data['Covs'] if use_covariates else None,
                data['predictor_names'],
                data['covariate_names'] if use_covariates else None,
                data['sample_id']
            )
            
            if predictions is None:
                print(f"Failed to make predictions for {target_id}")
                continue
            
            pred_values, pred_probas = predictions
                        
            # Save individual predictions
            result_df = pd.DataFrame({
                'sample_id': data['sample_id'],
                'prediction': pred_values
            })
            result_df.to_csv(os.path.join(output_dir, f"{target_id}_predictions.csv.gz"), index=False, compression="gzip")
            
            # Save probability predictions if available
            if pred_probas is not None:
                proba_cols = [f'class_{i}' for i in range(pred_probas.shape[1])]
                proba_df = pd.DataFrame(pred_probas, columns=proba_cols)
                proba_df.insert(0, 'sample_id', data['sample_id'])
                proba_df.to_csv(os.path.join(output_dir, f"{target_id}_probabilities.csv.gz"), index=False, compression="gzip")
            
            processed_count += 1
            print(f"Saved predictions for {target_id}")
        
        except Exception as e:
            print(f"Error making predictions for {model_file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully processed {processed_count} models")
    return

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing data files')
    parser.add_argument('--models_dir', type=str, default='./models', help='Directory containing model files')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save predictions')
    parser.add_argument('--force_covariates', action='store_true', help='Force using covariates for all models')
    parser.add_argument('--force_no_covariates', action='store_true', help='Force excluding covariates for all models')
    parser.add_argument('--force_scaling', action='store_true', help='Force using feature scaling for all models')
    parser.add_argument('--force_no_scaling', action='store_true', help='Force skipping feature scaling for all models')
    parser.add_argument('--no_skip_existing', action='store_true', help='Do not skip targets with existing predictions')
    parser.add_argument('--target_ids', type=str, nargs='+', help='Specific target IDs to process')
    
    args = parser.parse_args()
    
    # Determine whether to force covariates mode
    force_covs_mode = None
    if args.force_covariates and args.force_no_covariates:
        print("Warning: Both force_covariates and force_no_covariates specified. Using model-specific settings.")
    elif args.force_covariates:
        force_covs_mode = True
        print("Forcing all models to use covariates")
    elif args.force_no_covariates:
        force_covs_mode = False
        print("Forcing all models to not use covariates")
    else:
        print("Using model-specific covariate settings as defined in each model file")
    
    # Determine whether to force scaling mode
    force_scaling_mode = None
    if args.force_scaling and args.force_no_scaling:
        print("Warning: Both force_scaling and force_no_scaling specified. Using model-specific settings.")
    elif args.force_scaling:
        force_scaling_mode = True
        print("Forcing all models to use feature scaling")
    elif args.force_no_scaling:
        force_scaling_mode = False
        print("Forcing all models to not use feature scaling")
    else:
        print("Using model-specific scaling settings as defined in each model file")
    
    # If specific target IDs are provided, report them
    if args.target_ids:
        print(f"Processing only the following target IDs: {', '.join(args.target_ids)}")
    
    # Make predictions
    make_predictions(
        args.data_dir, 
        args.models_dir, 
        args.output_dir, 
        force_covs_mode,
        force_scaling_mode,
        skip_existing=not args.no_skip_existing,
        target_ids=args.target_ids
    )