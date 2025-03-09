"""
Module for loading and preprocessing data
"""
import numpy as np
import pandas as pd
import os

def load_data(data_dir='../data', include_covariates=False, prediction_only=False):
    """
    Load data from specified directory
    
    Parameters:
    -----------
    data_dir : str, default='../data'
        Directory containing data files
    include_covariates : bool, default=False
        Whether to include covariates in the data
    prediction_only : bool, default=False
        If True, only loads predictor data and optionally covariates, skipping targets
        
    Returns:
    --------
    dict
        Dictionary containing loaded data
    """
    data = {}
    
    # Load X.txt (predictors/proteins)
    try:
        X_df = pd.read_csv(f'{data_dir}/predictors.txt.gz', delimiter='\t')
        sample_id = X_df.iloc[:, 0]  # Get the first column as sample ID
        predictor_names = X_df.columns[1:]  # Save the predictor column names
        X_df = X_df.iloc[:, 1:]  # Remove sample ID column
        X = X_df.values  # Convert to NumPy array
        print(f"Loaded X data: {X.shape[1]} features, {X_df.shape[0]} samples")
        
        # Use column names directly as predictor IDs
        predictor_ids = predictor_names
        data['predictor_ids'] = predictor_ids
        
        data['X'] = X
        data['sample_id'] = sample_id
        data['predictor_names'] = predictor_names
            
    except Exception as e:
        print(f"Error loading predictors data: {e}")
        raise
    
    # Skip loading targets if prediction_only mode is enabled
    if not prediction_only:
        # Load Y.txt (targets)
        try:
            Y_df = pd.read_csv(f'{data_dir}/targets.txt.gz', delimiter='\t') 
            Y_sample_id = Y_df.iloc[:, 0]  # Get the sample ID column
            Y_df = Y_df.iloc[:, 1:]  # Remove sample ID column
            Y_annot_df = pd.read_csv(f'{data_dir}/target_annotation.txt', delimiter='\t')
            print(f"Loaded Y data: {Y_df.shape[1]} targets, {Y_df.shape[0]} samples")
            
            data['Y_df'] = Y_df
            data['Y_annot_df'] = Y_annot_df
            data['Y_sample_id'] = Y_sample_id
            
        except Exception as e:
            print(f"Error loading target data: {e}")
            raise
    else:
        # Create empty placeholders for Y data in prediction-only mode
        print("Prediction-only mode: Skipping target data loading")
        data['Y_df'] = pd.DataFrame()
        data['Y_annot_df'] = pd.DataFrame()
        data['Y_sample_id'] = None

    # Load Covs.txt (covariates) if covariates are to be included
    if include_covariates:
        try:
            Covs_df = pd.read_csv(f'{data_dir}/covs.txt.gz', delimiter='\t')
            Covs_sample_id = Covs_df.iloc[:, 0]  # Get the sample ID column
            covariate_names = Covs_df.columns[1:]  # Save the covariate column names
            Covs_df = Covs_df.iloc[:, 1:]  # Remove sample ID column
            Covs = Covs_df.values  # Convert to NumPy array
            Covs_annot_df = pd.read_csv(f'{data_dir}/cov_annotation.txt', delimiter='\t')
            print(f"Loaded covariate data: {Covs.shape[1]} covariates, {Covs.shape[0]} samples")
            
            data['Covs'] = Covs
            data['covariate_names'] = covariate_names
            data['Covs_annot_df'] = Covs_annot_df
            data['Covs_sample_id'] = Covs_sample_id
            
        except Exception as e:
            print(f"Error loading covariate data: {e}")
            raise
        
        # Verify sample IDs match across datasets (modified for prediction_only mode)
        if not prediction_only:
            try:
                assert np.all(sample_id == Y_sample_id), "Sample IDs don't match between X and Y"
                assert np.all(sample_id == Covs_sample_id), "Sample IDs don't match between X and Covs"
                print("Sample ID verification successful")
            except AssertionError as e:
                print(f"Error: {e}")
                raise
        else:
            # In prediction-only mode, only verify X and Covs match
            try:
                assert np.all(sample_id == Covs_sample_id), "Sample IDs don't match between X and Covs"
                print("Sample ID verification successful")
            except AssertionError as e:
                print(f"Error: {e}")
                raise
    else:
        # Create empty covariates if not including them
        Covs = np.zeros((X.shape[0], 0))
        covariate_names = np.array([])
        Covs_annot_df = pd.DataFrame()
        
        data['Covs'] = Covs
        data['covariate_names'] = covariate_names
        data['Covs_annot_df'] = Covs_annot_df
        
        # Verify sample IDs match between X and Y only (skip if prediction_only)
        if not prediction_only:
            try:
                assert np.all(sample_id == Y_sample_id), "Sample IDs don't match between X and Y"
                print("Sample ID verification successful")
            except AssertionError as e:
                print(f"Error: {e}")
                raise
    
    return data

def get_target_lists(Y_annot_df):
    """
    Get lists of targets based on target_group
    
    Parameters:
    -----------
    Y_annot_df : pandas.DataFrame
        Target annotation DataFrame
        
    Returns:
    --------
    dict
        Dictionary containing lists of targets by group
    """
    target_lists = {}
    
    # Define the targets to test based on target_group in Y_annot_df
    target_lists['longitudinal'] = Y_annot_df[Y_annot_df['target_group'] == "longitudinal"]['name'].values.tolist()
    target_lists['pathology'] = Y_annot_df[Y_annot_df['target_group'] == "pathology"]['name'].values.tolist()
    target_lists['omics'] = Y_annot_df[Y_annot_df['target_group'] == "omics"]['name'].values.tolist()
    target_lists['demographic'] = Y_annot_df[Y_annot_df['target_group'] == "demographic"]['name'].values.tolist()
    target_lists['genetic'] = Y_annot_df[Y_annot_df['target_group'] == "genetic"]['name'].values.tolist()
    target_lists['slope'] = Y_annot_df[Y_annot_df['target_group'] == "slope"]['name'].values.tolist()
    
    # Combine the target lists
    target_lists['all'] = (target_lists['longitudinal'] + 
                          target_lists['pathology'] + 
                          target_lists['omics'] + 
                          target_lists['demographic']+ 
                          target_lists['genetic']+
                          target_lists['slope'] )
    
    return target_lists

def create_output_directories(output_dir='.'):
    """
    Create directories for saving models, features, and results
    
    Parameters:
    -----------
    output_dir : str, default='.'
        Base output directory
    """
    # Original directories
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    
    # New directories for feature importance and predictions
    os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    print(f"Created output directories in {output_dir}")

def load_existing_result(target_id, output_dir='.'):
    """
    Load existing result from result file if it exists
    
    Parameters:
    -----------
    target_id : str
        Target ID
    output_dir : str, default='.'
        Output directory
        
    Returns:
    --------
    dict or None
        Result dictionary if it exists, None otherwise
    """
    result_file = os.path.join(output_dir, 'results', f"{target_id}_results.txt")
    
    if os.path.exists(result_file):
        try:
            import json
            with open(result_file, 'r') as f:
                result_dict = json.load(f)
            return result_dict
        except Exception as e:
            print(f"Error loading existing result for {target_id}: {e}")
    
    return None

def check_existing_results(target_id, output_dir='.'):
    """
    Check if results already exist for the given target_id
    
    Parameters:
    -----------
    target_id : str
        Target ID
    output_dir : str, default='.'
        Output directory
        
    Returns:
    --------
    bool
        True if results exist, False otherwise
    """
    result_file = os.path.join(output_dir, 'results', f"{target_id}_results.txt")
    model_file = os.path.join(output_dir, 'models', f"{target_id}_model.pkl")
    
    # Check if all files exist
    files_exist = os.path.exists(result_file) and os.path.exists(model_file)
    
    if files_exist:
        print(f"Results already exist for target ID {target_id}. Skipping.")
    
    return files_exist