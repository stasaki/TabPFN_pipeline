"""
Main execution script for model training
"""
import os
import time
import pandas as pd
import torch
import numpy as np
import argparse

from data_loader import load_data, create_output_directories, get_target_lists
from model_training import process_target

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train models on target variables')
    parser.add_argument('--include_covariates', action='store_true', help='Include covariates in models')
    parser.add_argument('--method', type=str, default='CatBoost', help='Feature selection method')
    parser.add_argument('--targets', type=str, nargs='+', default=[], help='Specific targets to process')
    parser.add_argument('--target_group', type=str, default='', choices=['longitudinal', 'pathology', 'omics', 'demographic', 'genetic', 'slope', 'all'], 
                       help='Target group to process (longitudinal, pathology, omics, demographic, genetic, slope, or all)')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory for output files')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    parser.add_argument('--scale_features', action='store_true', help='Apply StandardScaler to input features')
    args = parser.parse_args()
    
    # Print configuration
    print(f"Include covariates in models: {args.include_covariates}")
    print(f"Feature selection method: {args.method}")
    print(f"Scale input features: {args.scale_features}")
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
    
    # Load the data
    data = load_data(args.data_dir, args.include_covariates)
    
    # Pre-calculate how many features to select from X:
    # Final feature dimension = (selected features from X) + (number of covariates)
    d_covs = data['Covs'].shape[1]
    selected_k = 500 - d_covs if args.include_covariates else 500
    if selected_k <= 0:
        raise ValueError(f"Number of covariates ({d_covs}) is >= 500. Adjust your feature selection parameters.")
    print(f"Will select {selected_k} features from X to combine with {d_covs} covariates")
    
    # Get target lists
    target_lists = get_target_lists(data['Y_annot_df'])
    
    # Use specified targets or select from a target group
    if args.targets:
        test_targets = args.targets
    elif args.target_group:
        # If a target group is specified, use the corresponding list
        if args.target_group in target_lists:
            test_targets = target_lists[args.target_group]
            print(f"Using targets from group '{args.target_group}': {len(test_targets)} targets")
        else:
            raise ValueError(f"Unknown target group: {args.target_group}")
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
            not args.scale_features  # skip_scaling is True when scale_features is False
        )
        
        # If result is available, add it to the appropriate list
        if result_dict:
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
                "Include Covariates": r.get("Include Covariates", args.include_covariates),
                "Scale Features": args.scale_features,
                "Samples": r["Number of samples"],
                "Primary Metric": r["R2"],  # R2 as primary metric for regression
                "Time (s)": r["Time (s)"],
                "Model file": r["Model file"]
            }
            all_results.append(summ)

    if classification_results:
        for r in classification_results:
            summ = {
                "Target": r["Target"],
                "Target ID": r["Target ID"],
                "Type": "Classification",
                "Include Covariates": r.get("Include Covariates", args.include_covariates),
                "Scale Features": args.scale_features,
                "Samples": r["Number of samples"],
                "Primary Metric": r["Accuracy"],  # Accuracy as primary metric for classification
                "Time (s)": r["Time (s)"],
                "Model file": r["Model file"]
            }
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