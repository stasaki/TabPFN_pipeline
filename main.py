"""
Main execution script for model training
"""
import os
import time
import pandas as pd
import torch
import numpy as np
import argparse

from data_loader import load_data, create_output_directories, get_target_lists, get_predictor_groups, get_sample_groups
from model_training import process_target
from version import __version__

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train models on target variables')
    parser.add_argument('--include_covariates', action='store_true', help='Include covariates in models')
    parser.add_argument('--method', type=str, default='CatBoost', help='Feature selection method')
    parser.add_argument('--targets', type=str, nargs='+', default=[], help='Specific targets to process')
    parser.add_argument('--target_group', type=str, nargs='+', default=[], 
                       choices=['longitudinal', 'pathology', 'omics', 'demographic', 'genetic', 'slope', 'all'], 
                       help='Target groups to process (can specify multiple)')
    parser.add_argument('--predictor_group', type=str, nargs='+', default=[], 
                       help='Predictor groups to use (can specify multiple, "all" for all groups)')
    parser.add_argument('--test_sample_group', type=str, nargs='+', default=[],
                       help='Sample groups to use for testing (can specify multiple)')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory for output files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--scale_features', action='store_true', help='Apply StandardScaler to input features')
    parser.add_argument('--save_full_model', action='store_true', default=False, 
                       help='Save model trained on full data (default: True)')
    parser.add_argument('--save_train_model', action='store_true', 
                       help='Save model trained on training data only')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    args = parser.parse_args()
    
    # Print configuration
    print(f"Include covariates in models: {args.include_covariates}")
    print(f"Feature selection method: {args.method}")
    print(f"Scale input features: {args.scale_features}")
    print(f"Save model trained on full data: {args.save_full_model}")
    print(f"Save model trained on training data: {args.save_train_model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create directories for saving models and feature information
    create_output_directories(args.output_dir)
    
    # Check for GPU availability
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU")
    
    # If predictor_group is specified, validate predictor annotation exists first
    if args.predictor_group:
        # Check if the predictor annotation file exists
        predictor_annot_path = os.path.join(args.data_dir, 'predictor_annotation.txt')
        if not os.path.exists(predictor_annot_path):
            print(f"Warning: Predictor annotation file not found at {predictor_annot_path}")
            print("Cannot filter by predictor_group. Will use all predictors.")
            args.predictor_group = []
        else:
            print(f"Using predictor group(s): {', '.join(args.predictor_group)}")
    
    # Load the data with predictor group filtering if specified
    data = load_data(args.data_dir, args.include_covariates, predictor_group=args.predictor_group, 
                    sample_group=args.test_sample_group if args.test_sample_group else None)
    
    # If we successfully loaded predictor annotation, show available groups
    if 'predictor_annot_df' in data and not data['predictor_annot_df'].empty:
        available_predictor_groups = get_predictor_groups(data['predictor_annot_df'])
        print(f"Available predictor groups: {', '.join(available_predictor_groups)}")
    
    # If we successfully loaded sample annotation, show available groups
    if 'sample_annot_df' in data and not data['sample_annot_df'].empty:
        available_sample_groups = get_sample_groups(data['sample_annot_df'])
        print(f"Available sample groups: {', '.join(available_sample_groups)}")
        
        # If specific test sample groups were specified, show them
        if 'test_sample_groups' in data and data['test_sample_groups']:
            print(f"Using sample groups for testing: {', '.join(data['test_sample_groups'])}")
    
    # Pre-calculate how many features to select from X:
    # Final feature dimension = (selected features from X) + (number of covariates)
    d_covs = data['Covs'].shape[1]
    selected_k = 10 - d_covs if args.include_covariates else 10
    if selected_k <= 0:
        raise ValueError(f"Number of covariates ({d_covs}) is >= 10. Adjust your feature selection parameters.")
    print(f"Will select {selected_k} features from X to combine with {d_covs} covariates")
    
    # Get target lists
    target_lists = get_target_lists(data['Y_annot_df'])
    
    # Use specified targets or select from target groups
    if args.targets:
        test_targets = args.targets
    elif args.target_group:
        # Initialize an empty list to collect targets from all specified groups
        test_targets = []
        
        # Process each target group
        for group in args.target_group:
            if group in target_lists:
                # Add targets from this group to our collection
                group_targets = target_lists[group]
                test_targets.extend(group_targets)
                print(f"Added {len(group_targets)} targets from group '{group}'")
            elif group == 'all':
                # Special case: 'all' means all targets from all groups
                all_targets = []
                for g in ['longitudinal', 'pathology', 'omics', 'demographic', 'genetic', 'slope']:
                    if g in target_lists:
                        all_targets.extend(target_lists[g])
                test_targets.extend(all_targets)
                print(f"Added all {len(all_targets)} targets from all groups")
            else:
                raise ValueError(f"Unknown target group: {group}")
        
        # Remove duplicates (in case targets appear in multiple groups)
        test_targets = list(set(test_targets))
        print(f"Using a total of {len(test_targets)} unique targets from specified groups")
    else:
        # Default targets from original script
        test_targets = ["cts_mmse30", "msex", "age_at_visit"]
        print(f"Using default targets: {test_targets}")
        
    # Check if all specified targets exist
    missing_targets = [target for target in test_targets if target not in data['Y_df'].columns]
    if missing_targets:
        print(f"Warning: The following targets were not found in Y data: {missing_targets}")
        # Remove missing targets
        test_targets = [target for target in test_targets if target in data['Y_df'].columns]
        if not test_targets:
            raise ValueError("No valid targets to process.")
        print(f"Proceeding with valid targets: {test_targets}")
    
    # Prepare separate lists to store regression and classification results
    regression_results = []
    classification_results = []
    
    total_targets = len(test_targets)
    start_time = time.time()
    
    # Save predictor group information for documentation
    predictor_group_info = args.predictor_group if args.predictor_group else ["all"]
    
    # Save sample group information for documentation
    sample_group_info = args.test_sample_group if args.test_sample_group else ["random_split"]
    
    # Loop over each specified target
    for idx, target_name in enumerate(test_targets):
        # Show progress
        print(f"\nProcessing target {idx+1}/{total_targets}: {target_name}")
        
        # Process the target
        result_dict = process_target(
            data, 
            target_name, 
            args.include_covariates, 
            selected_k, 
            args.method, 
            args.verbose, 
            use_gpu,
            args.output_dir,
            not args.scale_features,  # skip_scaling is True when scale_features is False
            args.save_full_model,     # Pass option for saving the full model
            args.save_train_model     # Pass option for saving the training model
        )
        
        # If result is available, add it to the appropriate list and record predictor_group info
        if result_dict:
            # Add predictor group information
            result_dict["Predictor Groups"] = predictor_group_info
            
            # Add sample group information
            result_dict["Test Sample Groups"] = sample_group_info
            
            # Store result in appropriate list by type
            if result_dict["Type"] == "Regression":
                regression_results.append(result_dict)
            elif result_dict["Type"] == "Classification":
                classification_results.append(result_dict)
        
        # Estimate remaining time
        if idx < total_targets - 1:  # Only if there are more targets to process
            avg_time_per_target = (time.time() - start_time) / (idx+1)
            remaining_targets = total_targets - (idx+1)
            eta = remaining_targets * avg_time_per_target
            print(f"ETA for remaining targets: {eta:.2f} seconds")
    
    # Create summary reports
    # Convert the results lists to DataFrames
    if regression_results:
        reg_df = pd.DataFrame(regression_results)
        reg_df_sorted = reg_df.sort_values(by="R2", ascending=False)
        reg_df_sorted.to_csv(os.path.join(args.output_dir, "regression_results.csv"), index=False)
        print(f"\nProcessed {len(regression_results)} regression targets")
        print(f"Regression metrics saved to {os.path.join(args.output_dir, 'regression_results.csv')}")

    if classification_results:
        cls_df = pd.DataFrame(classification_results)
        cls_df_sorted = cls_df.sort_values(by="Accuracy", ascending=False)
        cls_df_sorted.to_csv(os.path.join(args.output_dir, "classification_results.csv"), index=False)
        print(f"\nProcessed {len(classification_results)} classification targets")
        print(f"Classification metrics saved to {os.path.join(args.output_dir, 'classification_results.csv')}")

    # Combine results for overall summary
    all_results = []
    if regression_results:
        for r in regression_results:
            summ = {
                "Target": r["Target"],
                "Target ID": r["Target ID"],
                "Type": "Regression",
                "Framework_Version": __version__,
                "Include Covariates": r.get("Include Covariates", args.include_covariates),
                "Scale Features": args.scale_features,
                "Predictor Groups": r.get("Predictor Groups", predictor_group_info),
                "Test Sample Groups": r.get("Test Sample Groups", sample_group_info),
                "Save Full Model": args.save_full_model,
                "Save Train Model": args.save_train_model,
                "Samples": r["Number of samples"],
                "Primary Metric": r["R2"],  # R2 as primary metric for regression
                "Time (s)": r["Time (s)"],
                "Model file": r["Model file"],
                "Feature importance file": r.get("Feature importance file", ""),
                "Predictions file": r.get("Predictions file", "")
            }
            
            # Add sampling information if present
            if "Training Sampling" in r:
                summ["Training Sampling"] = r["Training Sampling"]
            if "Test Sampling" in r:
                summ["Test Sampling"] = r["Test Sampling"]
                
            all_results.append(summ)

    if classification_results:
        for r in classification_results:
            summ = {
                "Target": r["Target"],
                "Target ID": r["Target ID"],
                "Type": "Classification",
                "Framework_Version": __version__,
                "Include Covariates": r.get("Include Covariates", args.include_covariates),
                "Scale Features": args.scale_features,
                "Predictor Groups": r.get("Predictor Groups", predictor_group_info),
                "Test Sample Groups": r.get("Test Sample Groups", sample_group_info),
                "Save Full Model": args.save_full_model,
                "Save Train Model": args.save_train_model,
                "Samples": r["Number of samples"],
                "Primary Metric": r["Accuracy"],  # Accuracy as primary metric for classification
                "Time (s)": r["Time (s)"],
                "Model file": r["Model file"],
                "Feature importance file": r.get("Feature importance file", ""),
                "Predictions file": r.get("Predictions file", "")
            }
            
            # Add sampling information if present
            if "Training Sampling" in r:
                summ["Training Sampling"] = r["Training Sampling"]
            if "Test Sampling" in r:
                summ["Test Sampling"] = r["Test Sampling"]
                
            all_results.append(summ)

    # Save combined results
    if all_results:
        all_df = pd.DataFrame(all_results)
        all_df.to_csv(os.path.join(args.output_dir, "all_results_summary.csv"), index=False)
        print(f"Combined summary saved to {os.path.join(args.output_dir, 'all_results_summary.csv')}")

    total_elapsed = time.time() - start_time
    print(f"\nAll targets processed in {total_elapsed:.2f} seconds.")
    print(f"Processed {len(all_results)} models")

if __name__ == "__main__":
    main()