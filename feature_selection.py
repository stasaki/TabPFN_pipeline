"""
Feature selection module that provides an ensemble feature selection approach
"""
import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectFromModel


class EnsembleFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that combines multiple feature selection methods:
    - Mutual Information
    - Tree-based feature importance (XGBoost, CatBoost, RandomForest)
    
    Features are ranked based on their aggregate importance across all methods,
    and the top k features are selected.
    
    Parameters:
    -----------
    k : int, default=10
        Number of top features to select overall.
    random_state : int, default=42
        Random seed for reproducibility.
    methods : list, default=['MI', 'XGB', 'CatBoost', 'RF']
        List of feature selection methods to use.
        Supported methods: 'MI' (Mutual Information), 'XGB' (XGBoost), 
        'CatBoost', 'RF' (RandomForest)
    verbose : int, default=0
        Verbosity level. Higher values mean more details.
        0: No output
        1: Basic information
        2: Detailed information including timing
    gpu : bool or dict, default=True
        Whether to use GPU acceleration for models that support it.
        If True, uses default GPU settings.
        If dict, can contain model-specific GPU settings:
        {
            'XGB': True,            # Enable GPU for XGBoost
            'CatBoost': {           # Specific settings for CatBoost
                'devices': '0',     # GPU device index
                'iterations': 100   # Reduced iterations for faster training
            }
        }
    task : str, default='regression'
        Type of task to perform. Options:
        - 'regression': For continuous target variables
        - 'classification': For categorical target variables
    """
    def __init__(self, k=10, random_state=42, methods=None, verbose=0, gpu=True, task='regression'):
        self.k = k
        self.random_state = random_state
        self.methods = methods if methods is not None else ['MI', 'XGB', 'CatBoost', 'RF']
        self.verbose = verbose
        self.gpu = gpu
        self.task = task
        self.selectors = []
        self.selected_features_ = None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.method_timing_ = {}
        self.total_time_ = None
        # Add a dictionary to store individual method importance scores
        self.method_importance_scores_ = {}
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values.
        **fit_params : dict
            Additional fit parameters.
            
        Returns:
        --------
        X_new : array-like of shape (n_samples, n_selected_features)
            Transformed array.
        """
        return self.fit(X, y, **fit_params).transform(X)
    
    def transform(self, X):
        """
        Reduce X to the selected features
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns:
        --------
        X_r : array-like of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        if not hasattr(self, 'selected_features_'):
            raise ValueError('EnsembleFeatureSelector has not been fitted yet.')
        
        # Check input dimensions
        if X.shape[1] != len(self.selected_features_):
            raise ValueError(f"X has {X.shape[1]} features, but EnsembleFeatureSelector "
                              f"is expecting {len(self.selected_features_)} features.")
        
        # Convert to numpy array if it's a DataFrame
        if hasattr(X, 'values'):
            X_arr = X.values
        else:
            X_arr = X
        
        # Select features using the boolean mask
        return X_arr[:, self.selected_features_]
    def fit(self, X, y, feature_names=None):
        import time
        start_time = time.time()
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names_ = np.array(feature_names)
        elif hasattr(X, 'columns'):
            self.feature_names_ = np.array(X.columns)
        else:
            self.feature_names_ = np.array([f"feature_{i}" for i in range(X.shape[1])])
            
        if self.verbose >= 1:
            print(f"Starting feature selection with {len(self.methods)} methods: {', '.join(self.methods)}")
            print(f"Target: select top {self.k} features out of {X.shape[1]} total features")
        
        available_methods = {
            'MI': self._create_mi_selector,
            'XGB': self._create_xgb_selector,
            'CatBoost': self._create_catboost_selector,
            'RF': self._create_rf_selector
        }
        
        # Validate selected methods
        for method in self.methods:
            if method not in available_methods:
                raise ValueError(f"Unknown feature selection method: {method}. "
                                f"Available methods: {list(available_methods.keys())}")
        
        # Initialize feature importance scores
        feature_importances = np.zeros(X.shape[1])
        
        # Create and fit selectors for each specified method
        for method in self.methods:
            if self.verbose >= 1:
                print(f"Fitting {method} selector...")
                
            method_start_time = time.time()
            
            # Get selector and feature importance scores
            selector, importance_scores = available_methods[method](X, y)
            # After getting importance_scores from each method
            self.selectors.append((method, selector))
            
            method_time = time.time() - method_start_time
            self.method_timing_[method] = method_time
            
            if self.verbose >= 2:
                print(f"  {method} completed in {method_time:.2f} seconds")
            
            # Store raw importance scores for each method
            self.method_importance_scores_[method] = importance_scores.copy()
            
            # Normalize importance scores to 0-1 range
            if importance_scores is not None:
                if np.sum(importance_scores) > 0:  # Avoid division by zero
                    normalized_scores = importance_scores / np.sum(importance_scores)
                    feature_importances += normalized_scores
                    
                    if self.verbose >= 2:
                        top_features_idx = np.argsort(importance_scores)[-5:][::-1]
                        print(f"  Top 5 features by {method}:")
                        for idx in top_features_idx:
                            print(f"    {self.feature_names_[idx]}: {importance_scores[idx]:.4f}")
        
        # Store feature importances - this is the combined importance from all methods
        self.feature_importances_ = feature_importances
        
        # Select top k features based on combined importance
        if self.k >= X.shape[1]:
            if self.verbose >= 1:
                print(f"Warning: k ({self.k}) >= number of features ({X.shape[1]}). Selecting all features.")
            self.selected_features_ = np.ones(X.shape[1], dtype=bool)
        else:
            # Get indices of top k features by importance
            top_indices = np.argsort(feature_importances)[-self.k:]
            self.selected_features_ = np.zeros(X.shape[1], dtype=bool)
            self.selected_features_[top_indices] = True
        
        self.total_time_ = time.time() - start_time
        
        if self.verbose >= 1:
            n_selected = np.sum(self.selected_features_)
            print(f"Selected {n_selected} features (target was {self.k})")
            print(f"Total feature selection time: {self.total_time_:.2f} seconds")
            
            if self.verbose >= 2:
                print("\nMethod timing breakdown:")
                for method, timing in self.method_timing_.items():
                    print(f"  {method}: {timing:.2f}s ({timing/self.total_time_*100:.1f}%)")
                    
                print("\nTop 10 selected features by combined importance:")
                combined_top_indices = np.argsort(feature_importances)[-10:][::-1]
                for idx in combined_top_indices:
                    if self.selected_features_[idx]:
                        status = "SELECTED"
                    else:
                        status = "not selected"
                    print(f"  {self.feature_names_[idx]}: {feature_importances[idx]:.4f} ({status})")
        
        return self
    
    def _create_mi_selector(self, X, y):
        """Create and fit Mutual Information selector based on task type"""
        if self.task == 'regression':
            # Calculate mutual information scores for regression
            mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
        else:
            # For classification, use mutual_info_classif
            from sklearn.feature_selection import mutual_info_classif
            mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            
        # Create a temporary selector to maintain compatibility
        mi_selector = SelectKBest(score_func=lambda X, y: mi_scores, k='all')
        mi_selector.fit(X, y)
        
        return mi_selector, mi_scores
    
    def _create_xgb_selector(self, X, y):
        """Create and fit XGBoost feature selector with optional GPU support based on task type"""
        # Configure GPU settings if enabled
        gpu_settings = {}
        if self.gpu:
            if isinstance(self.gpu, dict) and 'XGB' in self.gpu:
                gpu_config = self.gpu['XGB']
                if isinstance(gpu_config, dict):
                    # Use custom XGBoost GPU settings
                    gpu_settings = gpu_config
                elif gpu_config:  # If True
                    gpu_settings = {'tree_method': 'gpu_hist', 'gpu_id': 0}
            else:
                # Default GPU settings
                gpu_settings = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                
        if gpu_settings and self.verbose >= 1:
            print(f"  Training XGBoost ({self.task}) with GPU acceleration...")
            
        # Choose model based on task
        if self.task == 'regression':
            from xgboost import XGBRegressor
            xgb = XGBRegressor(
                random_state=self.random_state, 
                n_jobs=-1,
                **gpu_settings
            )
        else:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(
                random_state=self.random_state, 
                n_jobs=-1,
                **gpu_settings
            )
        
        try:
            xgb.fit(X, y)
            # Get feature importance scores
            importance_scores = xgb.feature_importances_
            
            if gpu_settings and self.verbose >= 1:
                print("  Successfully trained XGBoost on GPU")
                
        except Exception as e:
            if gpu_settings and self.verbose >= 1:
                print(f"  GPU training failed: {str(e)}")
                print("  Falling back to CPU training...")
                
            # Fall back to CPU
            if self.task == 'regression':
                from xgboost import XGBRegressor
                xgb = XGBRegressor(random_state=self.random_state, n_jobs=-1)
            else:
                from xgboost import XGBClassifier
                xgb = XGBClassifier(random_state=self.random_state, n_jobs=-1)
                
            xgb.fit(X, y)
            importance_scores = xgb.feature_importances_
        
        # Create a temporary selector for compatibility
        xgb_selector = SelectFromModel(xgb, prefit=True, threshold=-np.inf)
        
        return xgb_selector, importance_scores
    
    def _create_catboost_selector(self, X, y):
        """Create and fit CatBoost feature selector with optional GPU support based on task type"""
        # Configure GPU settings if enabled
        gpu_settings = {}
        if self.gpu:
            if isinstance(self.gpu, dict) and 'CatBoost' in self.gpu:
                gpu_config = self.gpu['CatBoost']
                if isinstance(gpu_config, dict):
                    # Use custom CatBoost GPU settings
                    gpu_settings = gpu_config
                    gpu_settings['task_type'] = 'GPU'
                elif gpu_config:  # If True
                    gpu_settings = {'task_type': 'GPU', 'devices': '0', 'iterations': 100}
            else:
                # Default GPU settings
                gpu_settings = {'task_type': 'GPU', 'devices': '0', 'iterations': 100}
                
        if gpu_settings and self.verbose >= 1:
            print(f"  Training CatBoost ({self.task}) with GPU acceleration...")
            
        # Choose model based on task
        if self.task == 'regression':
            from catboost import CatBoostRegressor
            catboost = CatBoostRegressor(
                random_state=self.random_state,
                verbose=0 if self.verbose < 2 else 10,
                train_dir=None, 
                **gpu_settings
            )
        else:
            from catboost import CatBoostClassifier
            catboost = CatBoostClassifier(
                random_state=self.random_state,
                verbose=0 if self.verbose < 2 else 10,
                train_dir=None, 
                **gpu_settings
            )
        
        try:
            catboost.fit(X, y)
            # Get feature importance scores
            importance_scores = catboost.feature_importances_
            
            if gpu_settings and self.verbose >= 1:
                print("  Successfully trained CatBoost on GPU")
                
        except Exception as e:
            if gpu_settings and self.verbose >= 1:
                print(f"  GPU training failed: {str(e)}")
                print("  Falling back to CPU training...")
                
            # Fall back to CPU
            if self.task == 'regression':
                from catboost import CatBoostRegressor
                catboost = CatBoostRegressor(random_state=self.random_state, verbose=0)
            else:
                from catboost import CatBoostClassifier
                catboost = CatBoostClassifier(random_state=self.random_state, verbose=0)
                
            catboost.fit(X, y)
            importance_scores = catboost.feature_importances_
        
        # Create a temporary selector for compatibility
        cat_selector = SelectFromModel(catboost, prefit=True, threshold=-np.inf)
        
        return cat_selector, importance_scores
    
    def _create_rf_selector(self, X, y):
        """Create and fit RandomForest feature selector based on task type"""
        if self.task == 'regression':
            from sklearn.ensemble import RandomForestRegressor
            # Train RandomForest model
            rf = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestClassifier
            # Train RandomForest model
            rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            
        rf.fit(X, y)
        
        # Get feature importance scores
        importance_scores = rf.feature_importances_
        
        # Create a temporary selector for compatibility
        rf_selector = SelectFromModel(rf, prefit=True, threshold=-np.inf)
        
        return rf_selector, importance_scores
    
    def summary(self):
        """
        Returns a detailed summary of the feature selection process.
        
        Returns:
        --------
        dict
            Dictionary containing summary information
        """
        if not hasattr(self, 'selected_features_'):
            return {"error": "Selector has not been fitted yet"}
        
        # Get indices of selected features
        selected_indices = np.where(self.selected_features_)[0]
        
        # Get feature names and their importance scores
        selected_features = []
        for idx in selected_indices:
            name = self.feature_names_[idx] if hasattr(self, 'feature_names_') else f"feature_{idx}"
            selected_features.append({
                "index": idx,
                "name": name,
                "importance": self.feature_importances_[idx]
            })
        
        # Sort by importance
        selected_features = sorted(selected_features, key=lambda x: x["importance"], reverse=True)
        
        # Create summary dictionary
        summary_dict = {
            "total_features": len(self.selected_features_),
            "selected_features_count": len(selected_indices),
            "target_k": self.k,
            "methods_used": self.methods,
            "selected_features": selected_features,
            "method_timing": self.method_timing_ if hasattr(self, 'method_timing_') else None,
            "total_time": self.total_time_ if hasattr(self, 'total_time_') else None
        }
        
        return summary_dict
    
    def transform(self, X):
        return X[:, self.selected_features_]
    
    def get_support(self, indices=False):
        if indices:
            return np.where(self.selected_features_)[0]
        return self.selected_features_
    
    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, 'selected_features_'):
            raise ValueError("Selector has not been fitted yet")
            
        if input_features is not None:
            feature_names = np.array(input_features)
        elif hasattr(self, 'feature_names_'):
            feature_names = self.feature_names_
        else:
            feature_names = np.array([f"feature_{i}" for i in range(len(self.selected_features_))])
            
        return feature_names[self.selected_features_]

    def save_feature_importance(self, output_dir, target_id):
        """
        Save feature importance scores to a gzipped CSV file
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the feature importance file
        target_id : str
            Target ID for file naming
        
        Returns:
        --------
        str
            Path to the saved file
        """
        import os
        import pandas as pd
        
        # Create features directory if it doesn't exist
        features_dir = os.path.join(output_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        
        # Create a DataFrame with feature names and combined importance scores
        importance_data = {
            'feature_name': self.feature_names_,
            'combined_importance': self.feature_importances_,
            'selected': self.selected_features_
        }
        
        # Add individual method importance scores to the DataFrame
        for method, scores in self.method_importance_scores_.items():
            importance_data[f'{method}_importance'] = scores
        
        importance_df = pd.DataFrame(importance_data)
        
        # Sort by combined importance score in descending order
        importance_df = importance_df.sort_values('combined_importance', ascending=False)
        
        # Save to gzipped CSV
        file_path = os.path.join(features_dir, f"{target_id}_feature_importance.csv.gz")
        importance_df.to_csv(file_path, index=False, compression='gzip')
        
        if self.verbose >= 1:
            print(f"Saved feature importance scores to {file_path} (gzip compressed)")
        
        return file_path



# Create a pipeline for X that scales and selects features using ensemble approach
def create_feature_selection_pipeline(selected_k=10, random_state=42, methods=None, verbose=0, gpu=True, task='regression', skip_scaling=False, output_dir=None, target_id=None):
    """
    Create a feature selection pipeline that applies scaling and ensemble feature selection.
    
    Parameters:
    -----------
    selected_k : int, default=10
        Number of top features to select overall.
    random_state : int, default=42
        Random seed for reproducibility.
    methods : list, default=None
        List of feature selection methods to use.
        Available methods: 'MI' (Mutual Information), 'XGB' (XGBoost), 
        'CatBoost', 'RF' (RandomForest)
        If None, uses all available methods.
    verbose : int, default=0
        Verbosity level:
        0: No output
        1: Basic information
        2: Detailed information
    gpu : bool or dict, default=True
        Whether to use GPU acceleration for models that support it.
        See EnsembleFeatureSelector documentation for details.
    task : str, default='regression'
        Type of task to perform. Options:
        - 'regression': For continuous target variables
        - 'classification': For categorical target variables
    skip_scaling : bool, default=False
        If True, skips the StandardScaler step, assuming data is already scaled.
    output_dir : str, default=None
        Directory to save feature importance information. If None, won't save.
    target_id : str, default=None
        Target ID for file naming when saving feature importance information.
        
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Pipeline containing StandardScaler (if skip_scaling=False) and EnsembleFeatureSelector
    """
    pipeline_steps = []
    
    # Add scaler only if we're not skipping scaling
    if not skip_scaling:
        pipeline_steps.append(('scaler', StandardScaler()))
    
    # Add feature selection
    feature_selector = EnsembleFeatureSelector(
        k=selected_k, 
        random_state=random_state,
        methods=methods,
        verbose=verbose,
        gpu=gpu,
        task=task
    )
    
    pipeline_steps.append(('feature_selection', feature_selector))
    
    pipeline_fs = Pipeline(pipeline_steps)
    
    return pipeline_fs