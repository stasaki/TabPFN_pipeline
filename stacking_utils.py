"""
Utilities for exploring and preparing stacking workflows
"""
import os
import glob
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict, Counter


def scan_cv_predictions(predictions_dir):
    """
    Scan directory for available CV prediction files
    
    Parameters:
    -----------
    predictions_dir : str
        Directory containing CV prediction files
        
    Returns:
    --------
    dict
        Dictionary with information about available predictions
    """
    cv_files = glob.glob(os.path.join(predictions_dir, "*_stacking_predictions.csv.gz"))
    
    if not cv_files:
        print(f"No CV prediction files (*_stacking_predictions.csv.gz) found in {predictions_dir}")
        return {}
    
    results = {}
    
    for file_path in cv_files:
        target_id = os.path.basename(file_path).replace("_stacking_predictions.csv.gz", "")
        
        try:
            df = pd.read_csv(file_path, compression='gzip')
            
            # Basic file info
            info = {
                'file_path': file_path,
                'total_rows': len(df),
                'has_fold_column': 'fold' in df.columns,
                'columns': list(df.columns)
            }
            
            # Check for CV vs missing target predictions
            if 'fold' in df.columns:
                cv_predictions = df[df['fold'] != 'full_model']
                missing_predictions = df[df['fold'] == 'full_model']
                
                info.update({
                    'cv_predictions': len(cv_predictions),
                    'missing_target_predictions': len(missing_predictions),
                    'unique_folds': sorted(df['fold'].unique()),
                    'samples_with_cv': cv_predictions['sample_id'].nunique() if len(cv_predictions) > 0 else 0
                })
            else:
                info.update({
                    'cv_predictions': len(df),
                    'missing_target_predictions': 0,
                    'unique_folds': ['unknown'],
                    'samples_with_cv': df['sample_id'].nunique() if 'sample_id' in df.columns else 0
                })
            
            # Determine target type
            if 'prediction' in df.columns:
                predictions = df['prediction'].dropna()
                if len(predictions) > 0:
                    unique_vals = predictions.nunique()
                    if unique_vals < 10 and all(isinstance(val, (int, np.integer)) or 
                                              (isinstance(val, float) and val.is_integer()) 
                                              for val in predictions.head(100)):
                        info['inferred_type'] = 'classification'
                    else:
                        info['inferred_type'] = 'regression'
                else:
                    info['inferred_type'] = 'unknown'
            
            # Check for probability columns (indicates classification)
            prob_cols = [col for col in df.columns if col.startswith('prob_class_')]
            if prob_cols:
                info['probability_columns'] = prob_cols
                info['inferred_type'] = 'classification'
            
            results[target_id] = info
            
        except Exception as e:
            results[target_id] = {'error': str(e)}
    
    return results


def analyze_stacking_potential(predictions_info, target_annot_df=None, min_features=3):
    """
    Analyze which targets are good candidates for stacking
    
    Parameters:
    -----------
    predictions_info : dict
        Information about available predictions from scan_cv_predictions()
    target_annot_df : pandas.DataFrame, optional
        Target annotation DataFrame for exclusion rules
    min_features : int, default=3
        Minimum number of features required for stacking
        
    Returns:
    --------
    dict
        Analysis results
    """
    # Filter to valid predictions
    valid_targets = {tid: info for tid, info in predictions_info.items() 
                    if 'error' not in info and info.get('cv_predictions', 0) > 0}
    
    if len(valid_targets) < 2:
        return {
            'message': f"Need at least 2 targets with valid CV predictions for stacking. Found {len(valid_targets)}.",
            'valid_targets': list(valid_targets.keys())
        }
    
    # Create exclusion rules if target annotation is available
    exclusion_rules = {}
    if target_annot_df is not None and not target_annot_df.empty:
        target_groups = target_annot_df.groupby('target_group')['target_id'].apply(list).to_dict()
        
        for target_id in valid_targets.keys():
            target_row = target_annot_df[target_annot_df['target_id'] == target_id]
            if not target_row.empty:
                target_group = target_row['target_group'].iloc[0]
                
                # Exclude same-group targets for certain categories
                exclude_same_group = ['omics', 'genetic', 'slope']
                
                if target_group in exclude_same_group:
                    exclusion_rules[target_id] = [tid for tid in target_groups.get(target_group, []) 
                                                 if tid != target_id and tid in valid_targets]
                else:
                    exclusion_rules[target_id] = []
    
    # Analyze each target
    analysis_results = {}
    
    for target_id, info in valid_targets.items():
        exclude_list = exclusion_rules.get(target_id, [])
        
        # Count available features (all other valid targets minus exclusions)
        available_features = len(valid_targets) - 1 - len(exclude_list)  # -1 for target itself
        
        analysis_results[target_id] = {
            'cv_predictions': info['cv_predictions'],
            'samples': info['samples_with_cv'],
            'inferred_type': info.get('inferred_type', 'unknown'),
            'excluded_targets': exclude_list,
            'available_features': available_features,
            'suitable_for_stacking': available_features >= min_features,
            'target_info': info
        }
    
    # Summary statistics
    suitable_targets = [tid for tid, analysis in analysis_results.items() 
                       if analysis['suitable_for_stacking']]
    
    type_counts = Counter(analysis['inferred_type'] for analysis in analysis_results.values())
    
    summary = {
        'total_targets_with_cv': len(valid_targets),
        'suitable_for_stacking': len(suitable_targets),
        'unsuitable_for_stacking': len(valid_targets) - len(suitable_targets),
        'target_type_distribution': dict(type_counts),
        'suitable_targets': suitable_targets,
        'min_features_threshold': min_features
    }
    
    return {
        'summary': summary,
        'target_analysis': analysis_results,
        'exclusion_rules': exclusion_rules
    }


def recommend_stacking_strategy(analysis_results):
    """
    Recommend stacking strategy based on analysis
    
    Parameters:
    -----------
    analysis_results : dict
        Results from analyze_stacking_potential()
        
    Returns:
    --------
    dict
        Recommendations
    """
    if 'summary' not in analysis_results:
        return {'error': 'Invalid analysis results'}
    
    summary = analysis_results['summary']
    target_analysis = analysis_results['target_analysis']
    
    recommendations = []
    
    # Overall feasibility
    if summary['suitable_for_stacking'] == 0:
        recommendations.append({
            'type': 'warning',
            'message': f"No targets suitable for stacking with current settings (min_features={summary['min_features_threshold']})",
            'suggestion': "Consider lowering min_features threshold or generating more CV predictions"
        })
        return {'recommendations': recommendations}
    
    # Batch vs individual strategy
    if summary['suitable_for_stacking'] > 10:
        recommendations.append({
            'type': 'strategy',
            'message': f"Large number of suitable targets ({summary['suitable_for_stacking']})",
            'suggestion': "Use batch_stacking.py for efficient processing"
        })
    elif summary['suitable_for_stacking'] > 1:
        recommendations.append({
            'type': 'strategy',
            'message': f"Moderate number of suitable targets ({summary['suitable_for_stacking']})",
            'suggestion': "Can use either stacking.py individually or batch_stacking.py"
        })
    else:
        recommendations.append({
            'type': 'strategy',
            'message': "Only one suitable target",
            'suggestion': "Use stacking.py for individual target processing"
        })
    
    # Target type recommendations
    reg_targets = [tid for tid, analysis in target_analysis.items() 
                  if analysis['suitable_for_stacking'] and analysis['inferred_type'] == 'regression']
    cls_targets = [tid for tid, analysis in target_analysis.items() 
                  if analysis['suitable_for_stacking'] and analysis['inferred_type'] == 'classification']
    
    if reg_targets and cls_targets:
        recommendations.append({
            'type': 'target_selection',
            'message': f"Mixed target types: {len(reg_targets)} regression, {len(cls_targets)} classification",
            'suggestion': "Consider processing regression and classification targets separately"
        })
    
    # Feature richness recommendations
    feature_counts = [analysis['available_features'] for analysis in target_analysis.values() 
                     if analysis['suitable_for_stacking']]
    
    if feature_counts:
        min_features = min(feature_counts)
        max_features = max(feature_counts)
        avg_features = np.mean(feature_counts)
        
        if min_features < 5:
            recommendations.append({
                'type': 'warning',
                'message': f"Some targets have few features (minimum: {min_features})",
                'suggestion': "Monitor these targets carefully - they may not benefit from stacking"
            })
        
        if avg_features > 20:
            recommendations.append({
                'type': 'optimization',
                'message': f"Rich feature environment (average: {avg_features:.1f} features per target)",
                'suggestion': "Consider feature selection within stacking models or ensemble methods"
            })
    
    # Priority recommendations
    high_sample_targets = [tid for tid, analysis in target_analysis.items() 
                          if analysis['suitable_for_stacking'] and analysis['samples'] > 1000]
    
    if high_sample_targets:
        recommendations.append({
            'type': 'priority',
            'message': f"High-sample targets available: {high_sample_targets[:5]}{'...' if len(high_sample_targets) > 5 else ''}",
            'suggestion': "Prioritize these targets for stable stacking model training"
        })
    
    return {'recommendations': recommendations}


def generate_stacking_commands(analysis_results, predictions_dir="./predictions", output_dir=".", data_dir="../data"):
    """
    Generate example command lines for stacking
    
    Parameters:
    -----------
    analysis_results : dict
        Results from analyze_stacking_potential()
    predictions_dir : str
        Predictions directory path
    output_dir : str
        Output directory path
    data_dir : str
        Data directory path
        
    Returns:
    --------
    list
        List of command line examples
    """
    if 'summary' not in analysis_results:
        return []
    
    summary = analysis_results['summary']
    suitable_targets = summary['suitable_targets']
    
    commands = []
    
    if not suitable_targets:
        return ["# No suitable targets found for stacking"]
    
    # Batch command
    if len(suitable_targets) > 1:
        commands.append("# Batch stacking for all suitable targets:")
        cmd = f"python batch_stacking.py \\\n"
        cmd += f"    --predictions_dir {predictions_dir} \\\n"
        cmd += f"    --output_dir {output_dir} \\\n"
        cmd += f"    --data_dir {data_dir} \\\n"
        cmd += f"    --cv_folds 5 \\\n"
        cmd += f"    --min_features {summary['min_features_threshold']} \\\n"
        cmd += f"    --gpu"
        commands.append(cmd)
        commands.append("")
    
    # Individual command examples
    example_targets = suitable_targets[:3]  # Show first 3 as examples
    
    commands.append("# Individual target stacking examples:")
    for target_id in example_targets:
        cmd = f"python stacking.py \\\n"
        cmd += f"    --predictions_dir {predictions_dir} \\\n"
        cmd += f"    --output_dir {output_dir} \\\n"
        cmd += f"    --target_variable {target_id} \\\n"
        cmd += f"    --cv_folds 5 \\\n"
        cmd += f"    --gpu"
        commands.append(cmd)
        commands.append("")
    
    # Analysis command
    commands.append("# Analyze stacking results:")
    cmd = f"python analyze_stacking_results.py \\\n"
    cmd += f"    --output_dir {output_dir} \\\n"
    cmd += f"    --create_plots"
    commands.append(cmd)
    
    return commands


def main():
    parser = argparse.ArgumentParser(description='Analyze CV predictions and plan stacking strategy')
    parser.add_argument('--predictions_dir', type=str, default='./predictions',
                       help='Directory containing CV prediction files')
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Data directory for target annotations')
    parser.add_argument('--min_features', type=int, default=3,
                       help='Minimum number of features required for stacking')
    parser.add_argument('--output_commands', action='store_true',
                       help='Output example stacking commands')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory (for command generation)')
    
    args = parser.parse_args()
    
    print("=== Stacking Strategy Analysis ===")
    print(f"Predictions directory: {args.predictions_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Minimum features threshold: {args.min_features}")
    print()
    
    # Scan available predictions
    print("Scanning for CV prediction files...")
    predictions_info = scan_cv_predictions(args.predictions_dir)
    
    if not predictions_info:
        print("No CV prediction files found. Ensure you have run training with --cv_folds parameter.")
        return
    
    print(f"Found {len(predictions_info)} prediction files")
    
    # Load target annotations if available
    target_annot_df = None
    annot_file = os.path.join(args.data_dir, 'target_annotation.txt')
    if os.path.exists(annot_file):
        try:
            target_annot_df = pd.read_csv(annot_file, delimiter='\t')
            print(f"Loaded target annotations: {len(target_annot_df)} targets")
        except Exception as e:
            print(f"Warning: Could not load target annotations: {e}")
    else:
        print("No target annotations found (optional)")
    
    print()
    
    # Analyze stacking potential
    print("Analyzing stacking potential...")
    analysis_results = analyze_stacking_potential(predictions_info, target_annot_df, args.min_features)
    
    if 'message' in analysis_results:
        print(analysis_results['message'])
        if 'valid_targets' in analysis_results:
            print(f"Valid targets: {analysis_results['valid_targets']}")
        return
    
    # Print summary
    summary = analysis_results['summary']
    print("=== SUMMARY ===")
    print(f"Targets with CV predictions: {summary['total_targets_with_cv']}")
    print(f"Suitable for stacking: {summary['suitable_for_stacking']}")
    print(f"Unsuitable (too few features): {summary['unsuitable_for_stacking']}")
    print(f"Target type distribution: {summary['target_type_distribution']}")
    print()
    
    # Print detailed analysis for unsuitable targets
    unsuitable_targets = [tid for tid, analysis in analysis_results['target_analysis'].items() 
                         if not analysis['suitable_for_stacking']]
    
    if unsuitable_targets:
        print("=== UNSUITABLE TARGETS ===")
        for target_id in unsuitable_targets:
            analysis = analysis_results['target_analysis'][target_id]
            print(f"{target_id}: {analysis['available_features']} features "
                  f"(need {args.min_features}+), {analysis['cv_predictions']} CV predictions")
        print()
    
    # Print recommendations
    recommendations = recommend_stacking_strategy(analysis_results)
    if 'recommendations' in recommendations:
        print("=== RECOMMENDATIONS ===")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"{i}. [{rec['type'].upper()}] {rec['message']}")
            print(f"   → {rec['suggestion']}")
        print()
    
    # Print example commands if requested
    if args.output_commands:
        print("=== EXAMPLE COMMANDS ===")
        commands = generate_stacking_commands(
            analysis_results, args.predictions_dir, args.output_dir, args.data_dir
        )
        for cmd in commands:
            print(cmd)
        print()
    
    # Feature matrix preview
    if summary['suitable_for_stacking'] > 0:
        print("=== FEATURE MATRIX PREVIEW ===")
        print("Available features for each target:")
        
        for target_id in summary['suitable_targets'][:10]:  # Show first 10
            analysis = analysis_results['target_analysis'][target_id]
            excluded = analysis['excluded_targets']
            available_features = analysis['available_features']
            
            exclusion_str = f" (excludes {len(excluded)} related)" if excluded else ""
            print(f"  {target_id}: {available_features} features{exclusion_str}")
        
        if len(summary['suitable_targets']) > 10:
            print(f"  ... and {len(summary['suitable_targets']) - 10} more targets")
    
    print("\nUse --output_commands to see example stacking commands")


if __name__ == "__main__":
    main()