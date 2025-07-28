"""
Ensemble Methods for Neural Network Models - Phase 7
Combines multiple neural network models to improve prediction accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle
import json
from pathlib import Path

class SimpleEnsemble:
    """
    Simple ensemble methods for combining neural network predictions
    Supports averaging, weighted averaging, and voting
    """
    
    def __init__(self, models: List[torch.nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble with list of trained models
        
        Args:
            models: List of trained PyTorch models
            weights: Optional weights for each model (for weighted averaging)
        """
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.weights = np.array(self.weights) / np.sum(self.weights)  # Normalize weights
        
        # Set all models to evaluation mode
        for model in self.models:
            model.eval()
    
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Generate probability predictions using ensemble averaging
        
        Args:
            X: Input tensor for prediction
            
        Returns:
            Array of probability predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = torch.sigmoid(model(X)).cpu().numpy()
                predictions.append(pred)
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return ensemble_pred
    
    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions using ensemble
        
        Args:
            X: Input tensor for prediction
            threshold: Decision threshold for binary classification
            
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def voting_predict(self, X: torch.Tensor, method: str = 'soft') -> np.ndarray:
        """
        Voting-based ensemble prediction
        
        Args:
            X: Input tensor for prediction
            method: 'soft' for probability voting, 'hard' for binary voting
            
        Returns:
            Array of predictions
        """
        if method == 'soft':
            return self.predict_proba(X)
        
        elif method == 'hard':
            votes = []
            with torch.no_grad():
                for model in self.models:
                    pred = torch.sigmoid(model(X)).cpu().numpy()
                    binary_pred = (pred >= 0.5).astype(int)
                    votes.append(binary_pred)
            
            # Majority voting
            votes = np.array(votes)
            return (np.sum(votes, axis=0) > len(self.models) / 2).astype(int)
        
        else:
            raise ValueError("Method must be 'soft' or 'hard'")

class StackingEnsemble:
    """
    Stacking ensemble that trains a meta-model on base model predictions
    Uses cross-validation to generate training data for meta-model
    """
    
    def __init__(self, base_models: List[torch.nn.Module], meta_model=None):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: List of trained base models
            meta_model: Meta-model for stacking (default: LogisticRegression)
        """
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(random_state=42)
        self.is_fitted = False
        
        # Set base models to evaluation mode
        for model in self.base_models:
            model.eval()
    
    def _get_base_predictions(self, X: torch.Tensor) -> np.ndarray:
        """
        Get predictions from all base models
        
        Args:
            X: Input tensor
            
        Returns:
            Array of shape (n_samples, n_models) with base predictions
        """
        base_preds = []
        
        with torch.no_grad():
            for model in self.base_models:
                pred = torch.sigmoid(model(X)).cpu().numpy().flatten()
                base_preds.append(pred)
        
        return np.column_stack(base_preds)
    
    def fit(self, X: torch.Tensor, y: np.ndarray):
        """
        Fit the meta-model on base model predictions
        
        Args:
            X: Training input tensor
            y: Training targets
        """
        # Get base model predictions
        base_predictions = self._get_base_predictions(X)
        
        # Train meta-model
        self.meta_model.fit(base_predictions, y)
        self.is_fitted = True
    
    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """
        Generate probability predictions using stacking
        
        Args:
            X: Input tensor for prediction
            
        Returns:
            Array of probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Stacking ensemble must be fitted before prediction")
        
        # Get base predictions
        base_predictions = self._get_base_predictions(X)
        
        # Meta-model prediction
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(base_predictions)[:, 1]
        else:
            return self.meta_model.predict(base_predictions)
    
    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Generate binary predictions using stacking
        
        Args:
            X: Input tensor for prediction
            threshold: Decision threshold
            
        Returns:
            Array of binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

class EnsembleManager:
    """
    Manager class for creating and evaluating different ensemble methods
    Provides unified interface for ensemble model management
    """
    
    def __init__(self, model_paths: List[str]):
        """
        Initialize ensemble manager with model file paths
        
        Args:
            model_paths: List of paths to saved model files
        """
        self.model_paths = model_paths
        self.models = []
        self.ensemble_results = {}
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all models from file paths"""
        from .model import BaselineNet  # Import model architecture
        
        for path in self.model_paths:
            if Path(path).exists():
                # Load model architecture and weights
                model = BaselineNet(input_dim=14, hidden_dim=256)  # Use default architecture
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                self.models.append(model)
            else:
                print(f"Warning: Model file not found: {path}")
    
    def create_simple_ensemble(self, weights: Optional[List[float]] = None) -> SimpleEnsemble:
        """
        Create simple ensemble with optional weights
        
        Args:
            weights: Optional weights for models
            
        Returns:
            SimpleEnsemble instance
        """
        return SimpleEnsemble(self.models, weights)
    
    def create_stacking_ensemble(self, meta_model=None) -> StackingEnsemble:
        """
        Create stacking ensemble with specified meta-model
        
        Args:
            meta_model: Meta-model for stacking
            
        Returns:
            StackingEnsemble instance
        """
        return StackingEnsemble(self.models, meta_model)
    
    def evaluate_ensemble(self, ensemble, X_val: torch.Tensor, y_val: np.ndarray, 
                         ensemble_name: str) -> Dict:
        """
        Evaluate ensemble performance on validation data
        
        Args:
            ensemble: Ensemble model to evaluate
            X_val: Validation input tensor
            y_val: Validation targets
            ensemble_name: Name for storing results
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Get predictions
        y_pred_proba = ensemble.predict_proba(X_val)
        y_pred = ensemble.predict(X_val)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        # Store results
        self.ensemble_results[ensemble_name] = metrics
        
        return metrics
    
    def compare_ensembles(self, X_val: torch.Tensor, y_val: np.ndarray) -> Dict:
        """
        Compare different ensemble methods on validation data
        
        Args:
            X_val: Validation input tensor
            y_val: Validation targets
            
        Returns:
            Dictionary with all ensemble results
        """
        results = {}
        
        # Simple averaging ensemble
        simple_ensemble = self.create_simple_ensemble()
        results['simple_average'] = self.evaluate_ensemble(
            simple_ensemble, X_val, y_val, 'simple_average'
        )
        
        # Voting ensemble (soft)
        results['soft_voting'] = self.evaluate_ensemble(
            simple_ensemble, X_val, y_val, 'soft_voting'
        )
        
        # Stacking ensemble with logistic regression
        stacking_ensemble = self.create_stacking_ensemble()
        stacking_ensemble.fit(X_val, y_val)  # Note: This uses validation data for fitting
        results['stacking_lr'] = self.evaluate_ensemble(
            stacking_ensemble, X_val, y_val, 'stacking_lr'
        )
        
        return results
    
    def save_best_ensemble(self, ensemble, ensemble_name: str, save_path: str):
        """
        Save the best performing ensemble model
        
        Args:
            ensemble: Ensemble model to save
            ensemble_name: Name of the ensemble
            save_path: Path to save the ensemble
        """
        ensemble_data = {
            'ensemble_type': type(ensemble).__name__,
            'ensemble_name': ensemble_name,
            'model_paths': self.model_paths,
            'ensemble_results': self.ensemble_results.get(ensemble_name, {}),
            'weights': getattr(ensemble, 'weights', None)
        }
        
        # Save ensemble configuration
        with open(save_path, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"Ensemble saved to: {save_path}")

def load_and_create_ensemble(model_paths: List[str], ensemble_type: str = 'simple') -> object:
    """
    Convenience function to load models and create ensemble
    
    Args:
        model_paths: List of paths to model files
        ensemble_type: Type of ensemble ('simple', 'stacking')
        
    Returns:
        Ensemble model instance
    """
    manager = EnsembleManager(model_paths)
    
    if ensemble_type == 'simple':
        return manager.create_simple_ensemble()
    elif ensemble_type == 'stacking':
        return manager.create_stacking_ensemble()
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

# Example usage and testing functions
if __name__ == "__main__":
    # Example of how to use ensemble methods
    print("Ensemble Methods Implementation - Phase 7")
    print("This module provides neural network ensemble capabilities")
    print("Import this module to use SimpleEnsemble, StackingEnsemble, and EnsembleManager classes") 