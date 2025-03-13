"""
Module for loading and preprocessing data
"""
import numpy as np
import pandas as pd
import os

def load_data(data_dir='../data', include_covariates=False, prediction_only=False, predictor_group=None):
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
    predictor_group : str or list, default=None
        Filter predictors to include only those in the specified group(s)
        If None, include all predictors
    sample_group : str or list, default=None
        Sample group(s) for test set selection
        If None, random split will be used
        
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
        predictor_names_all = X_df.columns[1:]  # Save all predictor column names
        X_df = X_df.iloc[:, 1:]  # Remove sample ID column

        
        # Load predictor annotation if it exists
        try:
            predictor_annot_df = pd.read_csv(f'{data_dir}/predictor_annotation.txt', delimiter='\t')
            print(f"Loaded predictor annotation: {predictor_annot_df.shape[0]} predictors annotated")
            data['predictor_annot_df'] = predictor_annot_df
        except Exception as e:
            print(f"Note: Could not load predictor annotation file: {e}")
            predictor_annot_df = None
            data['predictor_annot_df'] = pd.DataFrame()
        
        # Filter predictors by group if specified and annotation is available
        if predictor_group and predictor_annot_df is not None:
            # Convert single group to list for consistent handling
            if isinstance(predictor_group, str):
                predictor_group = [predictor_group]
                
            # Check if 'all' is in the predictor groups
            if 'all' in predictor_group:
                # Use all predictors
                print("Using all predictor groups")
                filtered_predictors = predictor_names_all
            else:
                # Get predictor names that match the specified group(s)
                mask = predictor_annot_df['predictor_group'].isin(predictor_group)
                filtered_predictor_names = predictor_annot_df.loc[mask, 'name']
                
                # Filter X to include only those predictors
                filtered_predictors = [col for col in predictor_names_all if col in filtered_predictor_names.values]
                print(f"Filtered to {len(filtered_predictors)} predictors from group(s): {', '.join(predictor_group)}")
                
                if len(filtered_predictors) == 0:
                    print("Warning: No predictors found in the specified group(s). Using all predictors.")
                    filtered_predictors = predictor_names_all
        else:
            # Use all predictors if no filtering is specified
            filtered_predictors = predictor_names_all
            
        # Filter X_df to include only the selected predictors
        X_df_filtered = X_df[filtered_predictors]
        X = X_df_filtered.values  # Convert to NumPy array
        predictor_names = np.array(filtered_predictors)
        
        print(f"Loaded X data: {X.shape[1]} features, {X_df_filtered.shape[0]} samples")
        
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

    # Load sample annotation file if it exists
    try:
        sample_annot_df = pd.read_csv(f'{data_dir}/sample_annotation.txt', delimiter='\t')
        print(f"Loaded sample annotation: {sample_annot_df.shape[0]} samples annotated")
        
        # Verify that sample_id is in the first column
        sample_id_col = sample_annot_df.columns[0]
        if sample_id_col != 'sample_id':
            print(f"Warning: First column in sample_annotation.txt is '{sample_id_col}', not 'sample_id'. Renaming.")
            sample_annot_df = sample_annot_df.rename(columns={sample_id_col: 'sample_id'})
        
        # Verify that all loaded sample IDs are in the annotation file
        missing_samples = set(sample_id) - set(sample_annot_df['sample_id'])
        if missing_samples:
            n_missing = len(missing_samples)
            print(f"Warning: {n_missing} samples in the data are not in sample_annotation.txt")
            if n_missing < 10:
                print(f"Missing samples: {list(missing_samples)}")
                
        data['sample_annot_df'] = sample_annot_df
        
        # Check if sample_group column exists in the annotation
        if 'sample_group' not in sample_annot_df.columns:
            print("Warning: 'sample_group' column not found in sample_annotation.txt. Group-based splits unavailable.")
        else:
            # Get unique sample groups
            unique_groups = sample_annot_df['sample_group'].unique()
            print(f"Available sample groups: {', '.join(unique_groups)}")
            
            # Store sample group information
            if sample_group:
                if isinstance(sample_group, str):
                    sample_group = [sample_group]
                
                # Validate that specified groups exist
                invalid_groups = set(sample_group) - set(unique_groups)
                if invalid_groups:
                    print(f"Warning: The following specified sample groups are not in the annotation: {invalid_groups}")
                
                valid_groups = set(sample_group) & set(unique_groups)
                if not valid_groups:
                    print("Warning: None of the specified sample groups are in the annotation. Using random split.")
                    data['test_sample_groups'] = None
                else:
                    print(f"Will use samples from group(s): {', '.join(valid_groups)} for testing")
                    data['test_sample_groups'] = list(valid_groups)
            else:
                data['test_sample_groups'] = None
                
    except Exception as e:
        print(f"Note: Could not load sample annotation file: {e}")
        data['sample_annot_df'] = pd.DataFrame()
        data['test_sample_groups'] = None
    
    return data

def get_samples_by_group(sample_ids, sample_annot_df, groups):
    """
    Get sample indices for specific groups
    
    Parameters:
    -----------
    sample_ids : array-like
        Sample IDs to filter
    sample_annot_df : pandas.DataFrame
        Sample annotation DataFrame
    groups : list
        List of group names to select
        
    Returns:
    --------
    numpy.ndarray
        Boolean mask indicating which samples belong to the specified groups
    """
    if sample_annot_df.empty or 'sample_group' not in sample_annot_df.columns or not groups:
        return None
    
    # Convert sample_ids to a pandas Series for easier comparison
    if not isinstance(sample_ids, pd.Series):
        sample_ids = pd.Series(sample_ids)
    
    # Filter annotation DataFrame to include only the samples in sample_ids
    valid_annot = sample_annot_df[sample_annot_df['sample_id'].isin(sample_ids)]
    
    # Get boolean mask for samples in the specified groups
    group_mask = valid_annot['sample_group'].isin(groups)
    
    # Create a mapping from sample ID to group membership
    sample_in_group = dict(zip(valid_annot['sample_id'], group_mask))
    
    # Map each sample ID to its group membership (True/False)
    # Default to False for any sample ID not in the annotation
    mask = sample_ids.map(lambda x: sample_in_group.get(x, False)).values
    
    return mask

def get_predictor_groups(predictor_annot_df):
    """
    Get unique predictor groups from predictor annotation
    
    Parameters:
    -----------
    predictor_annot_df : pandas.DataFrame
        Predictor annotation DataFrame
        
    Returns:
    --------
    list
        List of unique predictor groups
    """
    if predictor_annot_df is None or len(predictor_annot_df) == 0:
        return []
    
    return sorted(predictor_annot_df['predictor_group'].unique().tolist())

def get_sample_groups(sample_annot_df):
    """
    Get unique sample groups from sample annotation
    
    Parameters:
    -----------
    sample_annot_df : pandas.DataFrame
        Sample annotation DataFrame
        
    Returns:
    --------
    list
        List of unique sample groups
    """
    if sample_annot_df is None or len(sample_annot_df) == 0 or 'sample_group' not in sample_annot_df.columns:
        return []
    
    return sorted(sample_annot_df['sample_group'].unique().tolist())

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