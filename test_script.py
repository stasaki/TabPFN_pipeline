#!/usr/bin/env python
"""
Test script for sample grouping, TabPFN sampling, and CV stacking features.
This script tests:
1. Sample group-based test set selection
2. Handling of training datasets with more than 10,000 samples
3. Cross-validation mode for information leakage-free stacking predictions
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
from sklearn.datasets import make_regression, make_classification
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
# Note: Adjust these imports based on your actual project structure
from data_loader import create_output_directories, get_sample_groups, load_data
from model_training import process_target

def create_synthetic_data(n_train_samples=12000, n_test_samples=3000, n_features=1000, 
                         train_group='train', test_group='test', classification=False):
    """
    Create synthetic data for testing sample grouping and TabPFN sampling.
    
    Parameters:
    -----------
    n_train_samples : int, default=12000
        Number of training samples (> 10,000 to test TabPFN sampling)
    n_test_samples : int, default=3000
        Number of test samples
    n_features : int, default=1000
        Number of features
    train_group : str, default='train'
        Group to use for training
    test_group : str, default='test'
        Group to use for testing
    classification : bool, default=False
        Whether to generate classification data
    
    Returns:
    --------
    dict
        Dictionary containing synthetic data
    """
    n_samples = n_train_samples + n_test_samples
    print(f"Generating synthetic {'classification' if classification else 'regression'} data:")
    print(f"- {n_train_samples} training samples ('{train_group}' group)")
    print(f"- {n_test_samples} test samples ('{test_group}' group)")
    print(f"- {n_features} features")
    
    # Create output directories
    os.makedirs('test_data', exist_ok=True)
    
    # Create person IDs first (ensure we have enough unique groups for CV)
    # Make sure person IDs are unique between train and test
    train_person_prefix = "TRN"
    test_person_prefix = "TST"
    
    # Create enough unique person IDs to support CV folds
    # For CV to work, we need at least cv_folds unique person IDs
    # Let's create more person IDs by using smaller group sizes
    person_ids = []
    sample_ids = []
    
    for i in range(n_samples):
        if i < n_train_samples:
            # Training samples - create person IDs with smaller groups (3-5 samples per person)
            person_id = f"{train_person_prefix}{i//4:04d}"  # Every 4 samples get same person ID
            person_ids.append(person_id)
            # Sample ID format: personID_visitNumber
            visit_num = i % 4 + 1  # Visit numbers 1-4
            sample_ids.append(f"{person_id}_{visit_num:02d}")
        else:
            # Test samples
            test_idx = i - n_train_samples
            person_id = f"{test_person_prefix}{test_idx//4:04d}"  # Every 4 samples get same person ID
            person_ids.append(person_id) 
            # Sample ID format: personID_visitNumber
            visit_num = test_idx % 4 + 1  # Visit numbers 1-4
            sample_ids.append(f"{person_id}_{visit_num:02d}")
    
    # Assign groups - explicit assignment to ensure train group has >10,000 samples
    groups = np.array([train_group] * n_train_samples + [test_group] * n_test_samples)
    
    # Generate synthetic features
    if classification:
        # For classification, create binary target
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=20,
            n_redundant=10,
            n_classes=2,
            random_state=42
        )
        target_type = 'discrete'
        target_nlevels = 2
    else:
        # For regression, create continuous target
        X, y = make_regression(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=20,
            noise=0.1,
            random_state=42
        )
        target_type = 'continuous'
        target_nlevels = None
    
    # Create pandas DataFrames
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    X_df.insert(0, 'sample_id', sample_ids)  # Add sample_id as first column
    
    y_df = pd.DataFrame({
        'sample_id': sample_ids,
        'target_1': y
    })
    
    # Create sample annotation DataFrame
    sample_annotation = pd.DataFrame({
        'sample_id': sample_ids,
        'sample_group': groups,
        'person_id': person_ids
    })
    
    # Verify counts
    train_count = np.sum(groups == train_group)
    test_count = np.sum(groups == test_group)
    unique_person_ids = len(set(person_ids))
    print(f"Created {train_count} samples in train group and {test_count} samples in test group")
    print(f"Created {unique_person_ids} unique person IDs")
    print(f"Sample ID format examples: {sample_ids[:3]} ... {sample_ids[-3:]}")
    print(f"Person ID examples: {list(set(person_ids))[:5]} ...")
    
    # Create minimal target annotation DataFrame
    target_annotation = pd.DataFrame({
        'name': ['target_1'],
        'target_id': ['T001'],
        'type': [target_type],
        'nlevels': [target_nlevels],
        'target_group': ['test']  # Assign to 'test' group
    })
    
    # Create predictor annotation DataFrame (minimal)
    predictor_annotation = pd.DataFrame({
        'name': [f"feature_{i}" for i in range(n_features)],
        'predictor_group': ['test_group'] * n_features,
        'type': ['continuous'] * n_features
    })
    
    # Save files
    X_df.to_csv('test_data/predictors.txt.gz', sep='\t', index=False, compression='gzip')
    y_df.to_csv('test_data/targets.txt.gz', sep='\t', index=False, compression='gzip')
    sample_annotation.to_csv('test_data/sample_annotation.txt', sep='\t', index=False)
    target_annotation.to_csv('test_data/target_annotation.txt', sep='\t', index=False)
    predictor_annotation.to_csv('test_data/predictor_annotation.txt', sep='\t', index=False)
    
    # Create empty covariate files
    empty_covs = pd.DataFrame({'sample_id': sample_ids})
    empty_covs.to_csv('test_data/covs.txt.gz', sep='\t', index=False, compression='gzip')
    
    cov_annotation = pd.DataFrame({
        'name': [],
        'type': []
    })
    cov_annotation.to_csv('test_data/cov_annotation.txt', sep='\t', index=False)
    
    print(f"Saved test data files to 'test_data/' directory")
    
    # Create a data structure similar to what load_data() would return
    data = {
        'X': X,
        'Y_df': pd.DataFrame({'target_1': y}),
        'sample_id': pd.Series(sample_ids),
        'predictor_names': np.array([f"feature_{i}" for i in range(n_features)]),
        'Y_annot_df': target_annotation,
        'Covs': np.zeros((n_samples, 0)),  # Empty covariates
        'covariate_names': np.array([]),
        'Covs_annot_df': cov_annotation,
        'predictor_annot_df': predictor_annotation,
        'sample_annot_df': sample_annotation,
        'test_sample_groups': [test_group]
    }
    
    return data

def validate_cv_predictions(cv_predictions_file, expected_samples, cv_folds, target_type='regression'):
    """
    Validate CV predictions file for correctness
    
    Parameters:
    -----------
    cv_predictions_file : str
        Path to CV predictions file
    expected_samples : int
        Expected number of samples
    cv_folds : int
        Number of CV folds used
    target_type : str
        'regression' or 'classification'
    
    Returns:
    --------
    dict
        Validation results
    """
    print(f"\nValidating CV predictions file: {cv_predictions_file}")
    
    if not os.path.exists(cv_predictions_file):
        return {'valid': False, 'error': 'CV predictions file does not exist'}
    
    try:
        df = pd.read_csv(cv_predictions_file, compression='gzip')
        
        # Check required columns
        required_cols = ['sample_id', 'fold', 'set', 'true_value', 'prediction']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'valid': False, 'error': f'Missing columns: {missing_cols}'}
        
        # Check number of samples
        unique_samples = df['sample_id'].nunique()
        if unique_samples != expected_samples:
            return {'valid': False, 'error': f'Expected {expected_samples} unique samples, got {unique_samples}'}
        
        # Check that each sample appears exactly once
        sample_counts = df['sample_id'].value_counts()
        if not all(sample_counts == 1):
            return {'valid': False, 'error': 'Some samples appear more than once'}
        
        # Check fold distribution
        fold_counts = df['fold'].value_counts().sort_index()
        print(f"Fold distribution: {dict(fold_counts)}")
        
        # Check that all folds are represented
        expected_folds = set(range(1, cv_folds + 1))
        actual_folds = set(df['fold'].unique())
        if expected_folds != actual_folds:
            return {'valid': False, 'error': f'Expected folds {expected_folds}, got {actual_folds}'}
        
        # Check that all predictions are from test sets
        if not all(df['set'] == 'test'):
            return {'valid': False, 'error': 'CV predictions should all be from test sets'}
        
        # For classification, check for probability columns
        if target_type == 'classification':
            prob_cols = [col for col in df.columns if col.startswith('prob_class_')]
            print(f"Found {len(prob_cols)} probability columns: {prob_cols}")
        
        # Check for missing values
        missing_preds = df['prediction'].isna().sum()
        if missing_preds > 0:
            return {'valid': False, 'error': f'{missing_preds} missing predictions'}
        
        print(f"✓ CV predictions validation passed:")
        print(f"  - {unique_samples} unique samples")
        print(f"  - {len(df)} total predictions")
        print(f"  - {cv_folds} folds represented")
        print(f"  - All predictions from test sets")
        
        return {
            'valid': True, 
            'n_samples': unique_samples,
            'n_predictions': len(df),
            'n_folds': len(actual_folds),
            'fold_distribution': dict(fold_counts)
        }
        
    except Exception as e:
        return {'valid': False, 'error': f'Error reading CV predictions file: {str(e)}'}

def run_test(output_dir='test_output', data_dir='test_data', 
            n_train_samples=12000, n_test_samples=3000, n_features=1000, 
            train_group='train', test_group='test',
            classification=False, cv_folds=None, test_single_split=True):
    """
    Run a test of the sample grouping, TabPFN sampling, and CV stacking features.
    
    Parameters:
    -----------
    output_dir : str, default='test_output'
        Directory for test output
    data_dir : str, default='test_data'
        Directory for test data
    n_train_samples : int, default=12000
        Number of training samples (> 10,000 to test TabPFN sampling)
    n_test_samples : int, default=3000
        Number of test samples
    n_features : int, default=1000
        Number of features to generate
    train_group : str, default='train'
        Group to use for training
    test_group : str, default='test'
        Group to use for testing
    classification : bool, default=False
        Whether to test with classification data
    cv_folds : int, default=None
        Number of CV folds for stacking mode. If None, tests single split mode
    test_single_split : bool, default=True
        Whether to also test single split mode when CV mode is enabled
    """
    task_type = 'CLASSIFICATION' if classification else 'REGRESSION'
    mode_type = f'{cv_folds}-FOLD CV' if cv_folds else 'SINGLE SPLIT'
    
    print("\n" + "="*80)
    print(f"TESTING {task_type} WITH {mode_type} MODE")
    if cv_folds:
        print(f"Testing stacking predictions with {cv_folds} cross-validation folds")
    else:
        print("Testing sample grouping and >10K training samples")
    print("="*80)
    
    # Create output directories
    create_output_directories(output_dir)
    
    # Generate synthetic data
    data = create_synthetic_data(
        n_train_samples=n_train_samples, 
        n_test_samples=n_test_samples,
        n_features=n_features,
        train_group=train_group,
        test_group=test_group,
        classification=classification
    )
    
    # Display available sample groups
    available_groups = get_sample_groups(data['sample_annot_df'])
    print(f"Available sample groups: {', '.join(available_groups)}")
    
    if cv_folds:
        print(f"CV mode: Will use {cv_folds}-fold cross-validation (ignoring sample groups)")
        # Clear test sample groups for CV mode
        data['test_sample_groups'] = None
    else:
        print(f"Single split mode: Using '{test_group}' group for testing")
    
    # Process the target
    start_time = time.time()
    result = process_target(
        data=data,
        target_name='target_1',
        include_covariates=False,
        selected_k=20,  # Select fewer features for speed
        method="CatBoost",
        verbose=1,
        gpu=True,
        output_dir=output_dir,
        skip_scaling=False,
        save_full_model=True,
        save_train_model=True,
        compute_shap=False,  # Skip SHAP for faster testing
        cv_folds=cv_folds
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print(f"TEST COMPLETED IN {total_time:.2f} SECONDS")
    print("="*80)
    
    # Validate results based on mode
    if cv_folds:
        # CV mode validation
        print(f"\nCV MODE VALIDATION:")
        print(f"Expected CV folds: {cv_folds}")
        
        if result:
            print(f"CV Folds in result: {result.get('CV Folds', 'N/A')}")
            
            # Check for CV predictions file
            predictions_file = result.get('Predictions file', '')
            if predictions_file and '_cv_predictions.csv.gz' in predictions_file:
                print(f"✓ CV predictions file found: {predictions_file}")
                
                # Validate CV predictions
                validation_result = validate_cv_predictions(
                    predictions_file, 
                    n_train_samples + n_test_samples, 
                    cv_folds,
                    'classification' if classification else 'regression'
                )
                
                if validation_result['valid']:
                    print("✓ CV predictions validation PASSED")
                else:
                    print(f"✗ CV predictions validation FAILED: {validation_result['error']}")
            else:
                print(f"✗ CV predictions file not found or incorrectly named: {predictions_file}")
        else:
            print("✗ No result returned from CV processing")
            
        # Performance metrics for CV
        print("\nCV PERFORMANCE METRICS:")
        if result and classification:
            print(f"  Average Accuracy: {result.get('Accuracy', 'N/A')}")
            print(f"  Average F1 Score: {result.get('F1', 'N/A')}")
        elif result:
            print(f"  Average R2 Score: {result.get('R2', 'N/A')}")
            print(f"  Average MSE: {result.get('MSE', 'N/A')}")
            
    else:
        # Single split mode validation
        print(f"\nSINGLE SPLIT MODE VALIDATION:")
        
        # Check if result contains sampling information
        if result and 'Training Sampling' in result:
            print("\nTraining Sampling Info:")
            for k, v in result['Training Sampling'].items():
                print(f"  {k}: {v}")
        else:
            print("\nNo Training Sampling information found")
                
        if result and 'Test Sampling' in result:
            print("\nTest Sampling Info:")
            for k, v in result['Test Sampling'].items():
                print(f"  {k}: {v}")
        
        # Performance metrics for single split
        print("\nSINGLE SPLIT PERFORMANCE METRICS:")
        if result and classification:
            print(f"  Accuracy: {result.get('Accuracy', 'N/A')}")
            print(f"  F1 Score: {result.get('F1', 'N/A')}")
        elif result:
            print(f"  R2 Score: {result.get('R2', 'N/A')}")
            print(f"  MSE: {result.get('MSE', 'N/A')}")
    
    # Test both modes if requested
    if cv_folds and test_single_split:
        print(f"\n" + "-"*60)
        print("TESTING SINGLE SPLIT MODE FOR COMPARISON")
        print("-"*60)
        
        # Run single split test
        single_split_result = run_test(
            output_dir=output_dir + '_single',
            data_dir=data_dir,
            n_train_samples=n_train_samples,
            n_test_samples=n_test_samples,
            n_features=n_features,
            train_group=train_group,
            test_group=test_group,
            classification=classification,
            cv_folds=None,
            test_single_split=False  # Avoid infinite recursion
        )
        
        print("\nMODE COMPARISON:")
        if result and single_split_result:
            if classification:
                cv_acc = result.get('Accuracy', 0)
                single_acc = single_split_result.get('Accuracy', 0)
                print(f"CV Average Accuracy: {cv_acc:.4f}")
                print(f"Single Split Accuracy: {single_acc:.4f}")
            else:
                cv_r2 = result.get('R2', 0)
                single_r2 = single_split_result.get('R2', 0)
                print(f"CV Average R2: {cv_r2:.4f}")
                print(f"Single Split R2: {single_r2:.4f}")
    
    print(f"\nTest results: {'SUCCESS' if result else 'FAILURE'}")
    
    return result

def run_comprehensive_test():
    """
    Run comprehensive tests covering multiple scenarios
    """
    print("="*80)
    print("RUNNING COMPREHENSIVE TESTS")
    print("="*80)
    
    test_scenarios = [
        # (classification, cv_folds, description)
        (False, None, "Regression with single split"),
        (False, 3, "Regression with 3-fold CV"),
        (True, None, "Classification with single split"),
        (True, 5, "Classification with 5-fold CV"),
    ]
    
    results = {}
    
    for classification, cv_folds, description in test_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {description}")
        print(f"{'='*60}")
        
        scenario_key = f"{'cls' if classification else 'reg'}_{'cv' + str(cv_folds) if cv_folds else 'single'}"
        
        try:
            result = run_test(
                output_dir=f'test_output_{scenario_key}',
                n_train_samples=1000,  # Smaller for faster testing
                n_test_samples=300,
                n_features=100,       # Fewer features for faster testing
                classification=classification,
                cv_folds=cv_folds,
                test_single_split=False  # Skip comparison mode for comprehensive test
            )
            
            results[scenario_key] = {
                'success': result is not None,
                'result': result,
                'description': description
            }
            
        except Exception as e:
            print(f"ERROR in scenario {description}: {str(e)}")
            results[scenario_key] = {
                'success': False,
                'error': str(e),
                'description': description
            }
    
    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    for scenario_key, result_info in results.items():
        status = "✓ PASS" if result_info['success'] else "✗ FAIL"
        print(f"{status}: {result_info['description']}")
        if not result_info['success'] and 'error' in result_info:
            print(f"     Error: {result_info['error']}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test sample grouping, TabPFN sampling, and CV stacking')
    parser.add_argument('--output_dir', type=str, default='test_output', help='Output directory')
    parser.add_argument('--data_dir', type=str, default='test_data', help='Data directory')
    parser.add_argument('--n_train_samples', type=int, default=12000, help='Number of training samples')
    parser.add_argument('--n_test_samples', type=int, default=3000, help='Number of test samples')
    parser.add_argument('--n_features', type=int, default=1000, help='Number of features')
    parser.add_argument('--train_group', type=str, default='train', help='Group to use for training')
    parser.add_argument('--test_group', type=str, default='test', help='Group to use for testing')
    parser.add_argument('--classification', action='store_true', help='Use classification instead of regression')
    parser.add_argument('--cv_folds', type=int, default=None, help='Number of CV folds for stacking mode')
    parser.add_argument('--comprehensive', action='store_true', help='Run comprehensive tests for multiple scenarios')
    parser.add_argument('--no_single_split_comparison', action='store_true', help='Skip single split comparison when testing CV mode')
    
    args = parser.parse_args()
    
    if args.comprehensive:
        results = run_comprehensive_test()
    else:
        result = run_test(
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            n_train_samples=args.n_train_samples,
            n_test_samples=args.n_test_samples,
            n_features=args.n_features,
            train_group=args.train_group,
            test_group=args.test_group,
            classification=args.classification,
            cv_folds=args.cv_folds,
            test_single_split=not args.no_single_split_comparison
        )