"""
Tree-Based Models Implementation - Phase 7
XGBoost, LightGBM, and Random Forest models optimized for structured data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from pathlib import Path
import time

# Tree-based model imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class XGBoostModel:
    """
    XGBoost implementation with hyperparameter optimization
    Optimized for binary classification on structured data
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize XGBoost model with default parameters
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.feature_importance = None
        
        # Default parameters optimized for binary classification
        self.default_params = {
            'objective': 'binary:logistic',
            'random_state': random_state,
            'eval_metric': 'logloss',
            'verbosity': 0
        }
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using randomized search with validation set
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of hyperparameter combinations to try
            
        Returns:
            Dictionary with best parameters and results
        """
        print("🔍 Starting XGBoost hyperparameter optimization...")
        start_time = time.time()
        
        # Parameter search space
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(**self.default_params)
        
        # Randomized search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        # Fit the search
        random_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = random_search.best_params_
        self.best_params.update(self.default_params)
        
        # Train final model with best parameters
        self.model = xgb.XGBClassifier(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        results = {
            'best_params': self.best_params,
            'cv_score': random_search.best_score_,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_pred_proba),
            'optimization_time': time.time() - start_time
        }
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ XGBoost optimization complete in {results['optimization_time']:.2f}s")
        print(f"🎯 Best CV Score: {results['cv_score']:.4f}")
        print(f"📊 Validation Accuracy: {results['val_accuracy']:.4f}")
        
        return results
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train XGBoost model with optional early stopping
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
        """
        if self.best_params is None:
            # Use default parameters if not optimized
            params = self.default_params.copy()
            params.update({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1
            })
        else:
            params = self.best_params
        
        self.model = xgb.XGBClassifier(**params)
        
        # Fit with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance as DataFrame
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

class LightGBMModel:
    """
    LightGBM implementation with hyperparameter optimization
    Optimized for memory efficiency and speed
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize LightGBM model with default parameters
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.training_history = {}
        self.feature_importance = None
        
        # Default parameters optimized for binary classification
        self.default_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': random_state,
            'verbosity': -1,
            'force_col_wise': True
        }
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials: int = 50) -> Dict:
        """
        Optimize hyperparameters using randomized search
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of hyperparameter combinations to try
            
        Returns:
            Dictionary with best parameters and results
        """
        print("🔍 Starting LightGBM hyperparameter optimization...")
        start_time = time.time()
        
        # Parameter search space
        param_grid = {
            'n_estimators': [100, 200, 300, 500, 800, 1000],
            'max_depth': [3, 4, 5, 6, 7, 8, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_samples': [5, 10, 20, 30, 50],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
            'num_leaves': [15, 31, 63, 127, 255]
        }
        
        # Create base model
        base_model = lgb.LGBMClassifier(**self.default_params)
        
        # Randomized search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        # Fit the search
        random_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = random_search.best_params_
        self.best_params.update(self.default_params)
        
        # Train final model with best parameters
        self.model = lgb.LGBMClassifier(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        results = {
            'best_params': self.best_params,
            'cv_score': random_search.best_score_,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_pred_proba),
            'optimization_time': time.time() - start_time
        }
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ LightGBM optimization complete in {results['optimization_time']:.2f}s")
        print(f"🎯 Best CV Score: {results['cv_score']:.4f}")
        print(f"📊 Validation Accuracy: {results['val_accuracy']:.4f}")
        
        return results
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        Train LightGBM model with optional early stopping
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping
        """
        if self.best_params is None:
            # Use default parameters if not optimized
            params = self.default_params.copy()
            params.update({
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31
            })
        else:
            params = self.best_params
        
        self.model = lgb.LGBMClassifier(**params)
        
        # Fit with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

class RandomForestModel:
    """
    Random Forest implementation with hyperparameter optimization
    Robust baseline ensemble method
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Random Forest model
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
        # Default parameters
        self.default_params = {
            'random_state': random_state,
            'n_jobs': -1
        }
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                n_trials: int = 30) -> Dict:
        """
        Optimize hyperparameters using randomized search
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of hyperparameter combinations to try
            
        Returns:
            Dictionary with best parameters and results
        """
        print("🔍 Starting Random Forest hyperparameter optimization...")
        start_time = time.time()
        
        # Parameter search space
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7, None],
            'bootstrap': [True, False],
            'class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
        # Create base model
        base_model = RandomForestClassifier(**self.default_params)
        
        # Randomized search with cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_trials,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        
        # Fit the search
        random_search.fit(X_train, y_train)
        
        # Store best parameters
        self.best_params = random_search.best_params_
        self.best_params.update(self.default_params)
        
        # Train final model with best parameters
        self.model = RandomForestClassifier(**self.best_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        results = {
            'best_params': self.best_params,
            'cv_score': random_search.best_score_,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_pred_proba),
            'optimization_time': time.time() - start_time
        }
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ Random Forest optimization complete in {results['optimization_time']:.2f}s")
        print(f"🎯 Best CV Score: {results['cv_score']:.4f}")
        print(f"📊 Validation Accuracy: {results['val_accuracy']:.4f}")
        
        return results
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        if self.best_params is None:
            # Use default parameters if not optimized
            params = self.default_params.copy()
            params.update({
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            })
        else:
            params = self.best_params
        
        self.model = RandomForestClassifier(**params)
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate binary predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance as DataFrame"""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

class TreeModelManager:
    """
    Manager class for training and comparing all tree-based models
    Provides unified interface for model management and comparison
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize tree model manager
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
        # Initialize all models
        self.models['xgboost'] = XGBoostModel(random_state)
        self.models['lightgbm'] = LightGBMModel(random_state)
        self.models['random_forest'] = RandomForestModel(random_state)
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        optimize_hyperparams: bool = True,
                        n_trials: int = 50) -> Dict:
        """
        Train all tree-based models with optional hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of trials for hyperparameter optimization
            
        Returns:
            Dictionary with all model results
        """
        print("🚀 Starting tree-based model training...")
        start_time = time.time()
        
        all_results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Training {model_name.upper()}...")
            
            try:
                if optimize_hyperparams:
                    # Optimize hyperparameters
                    results = model.optimize_hyperparameters(
                        X_train, y_train, X_val, y_val, n_trials
                    )
                else:
                    # Train with default parameters
                    model.fit(X_train, y_train, X_val, y_val)
                    
                    # Evaluate on validation set
                    val_pred = model.predict(X_val)
                    val_pred_proba = model.predict_proba(X_val)
                    
                    results = {
                        'val_accuracy': accuracy_score(y_val, val_pred),
                        'val_f1': f1_score(y_val, val_pred),
                        'val_precision': precision_score(y_val, val_pred),
                        'val_recall': recall_score(y_val, val_pred),
                        'val_roc_auc': roc_auc_score(y_val, val_pred_proba)
                    }
                
                all_results[model_name] = results
                self.results[model_name] = results
                
                print(f"✅ {model_name.upper()} completed - Accuracy: {results['val_accuracy']:.4f}")
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        total_time = time.time() - start_time
        print(f"\n🏁 All models trained in {total_time:.2f}s")
        
        # Find best model
        best_model = max(
            [(name, res) for name, res in all_results.items() if 'val_accuracy' in res],
            key=lambda x: x[1]['val_accuracy']
        )
        
        print(f"🏆 Best model: {best_model[0].upper()} ({best_model[1]['val_accuracy']:.4f} accuracy)")
        
        return all_results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model
        
        Returns:
            Tuple of (model_name, model_instance)
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        best_name = max(
            self.results.keys(),
            key=lambda name: self.results[name].get('val_accuracy', 0)
        )
        
        return best_name, self.models[best_name]
    
    def save_models(self, save_dir: str = 'models'):
        """
        Save all trained models
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            if model.model is not None:
                model_file = save_path / f'{model_name}_model.pkl'
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                print(f"💾 Saved {model_name} to {model_file}")
        
        # Save results summary
        results_file = save_path / 'tree_models_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📊 Saved results to {results_file}")
    
    def compare_models(self) -> pd.DataFrame:
        """
        Create comparison DataFrame of all model results
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No models have been trained yet")
        
        comparison_data = []
        for model_name, results in self.results.items():
            if 'val_accuracy' in results:
                comparison_data.append({
                    'Model': model_name.title(),
                    'Accuracy': results['val_accuracy'],
                    'F1 Score': results['val_f1'],
                    'Precision': results['val_precision'],
                    'Recall': results['val_recall'],
                    'ROC AUC': results['val_roc_auc']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
        
        return comparison_df

# Example usage and utility functions
def quick_tree_model_comparison(X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               optimize: bool = True) -> Dict:
    """
    Quick function to train and compare all tree models
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        optimize: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with all results
    """
    manager = TreeModelManager()
    results = manager.train_all_models(X_train, y_train, X_val, y_val, optimize)
    
    # Print comparison
    if any('val_accuracy' in res for res in results.values()):
        comparison_df = manager.compare_models()
        print("\n📊 Tree Model Comparison:")
        print(comparison_df.round(4))
    
    return results

if __name__ == "__main__":
    print("Tree-Based Models Implementation - Phase 7")
    print("This module provides XGBoost, LightGBM, and Random Forest implementations")
    print("Use TreeModelManager for comprehensive model training and comparison") 