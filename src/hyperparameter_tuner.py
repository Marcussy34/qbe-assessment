"""
Hyperparameter tuning module for systematic optimization.
Implements grid search and random search for finding optimal configurations.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime
import json
import os

from utils import set_random_seeds, get_device
from data_loader import load_pickle_data
from preprocessor import DataPreprocessor, create_data_splits, create_data_loaders
from model import create_model
from train import ModelTrainer
from evaluate import ModelEvaluator


class HyperparameterTuner:
    """
    Systematic hyperparameter tuning using grid search and random search.
    Optimizes model architecture and training parameters to maximize accuracy.
    """
    
    def __init__(self, X_train, X_val, y_train, y_val, device=None, random_state=42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            X_train, X_val: Training and validation features
            y_train, y_val: Training and validation targets
            device: PyTorch device (GPU/CPU)
            random_state: Random seed for reproducibility
        """
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.device = device or get_device()
        self.random_state = random_state
        
        # Results storage
        self.results = []
        self.best_config = None
        self.best_score = 0.0
        
        print(f"Hyperparameter Tuner initialized:")
        print(f"  Training samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Device: {self.device}")
        print(f"  Input features: {X_train.shape[1]}")
    
    def define_search_space(self):
        """
        Define hyperparameter search space for optimization.
        
        Returns:
            dict: Hyperparameter search space configuration
        """
        search_space = {
            # Model architecture
            'model_type': ['baseline', 'optimized', 'deep', 'wide'],
            
            # Baseline model parameters
            'hidden_dim': [32, 64, 128, 256],
            'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            
            # Optimized model parameters
            'hidden_dims': [
                [64, 32],
                [128, 64], 
                [128, 64, 32],
                [256, 128, 64],
                [512, 256, 128]
            ],
            'use_batch_norm': [True, False],
            'use_residual': [True, False],
            
            # Training parameters
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'optimizer_type': ['adam', 'adamw', 'sgd'],
            'scheduler_type': ['plateau', 'step', 'cosine'],
            'batch_size': [32, 64, 128, 256],
            
            # Training strategy
            'num_epochs': [20, 50, 100],
            'early_stopping_patience': [5, 10, 15],
            'weight_decay': [0.0, 1e-5, 1e-4, 1e-3]
        }
        
        return search_space
    
    def generate_baseline_configs(self, n_configs=20):
        """
        Generate configurations focused on baseline model optimization.
        
        Args:
            n_configs (int): Number of configurations to generate
            
        Returns:
            list: List of baseline model configurations
        """
        configs = []
        
        # Grid search for baseline model
        hidden_dims = [32, 64, 128, 256]
        dropout_rates = [0.1, 0.2, 0.3, 0.4]
        learning_rates = [0.0001, 0.001, 0.01]
        optimizers = ['adam', 'adamw']
        
        for hidden_dim, dropout, lr, opt in product(hidden_dims, dropout_rates, learning_rates, optimizers):
            config = {
                'model_type': 'baseline',
                'hidden_dim': hidden_dim,
                'dropout_rate': dropout,
                'learning_rate': lr,
                'optimizer_type': opt,
                'scheduler_type': 'plateau',
                'batch_size': 64,
                'num_epochs': 50,
                'early_stopping_patience': 10,
                'weight_decay': 0.0 if opt == 'adam' else 1e-4
            }
            configs.append(config)
            
            if len(configs) >= n_configs:
                break
        
        return configs[:n_configs]
    
    def generate_advanced_configs(self, n_configs=15):
        """
        Generate configurations for advanced architectures.
        
        Args:
            n_configs (int): Number of configurations to generate
            
        Returns:
            list: List of advanced model configurations
        """
        configs = []
        
        # Optimized model configs
        optimized_configs = [
            {
                'model_type': 'optimized',
                'hidden_dims': [128, 64, 32],
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'use_residual': False,
                'learning_rate': 0.001,
                'optimizer_type': 'adam',
                'scheduler_type': 'plateau'
            },
            {
                'model_type': 'optimized', 
                'hidden_dims': [256, 128, 64],
                'dropout_rate': 0.2,
                'use_batch_norm': True,
                'use_residual': True,
                'learning_rate': 0.0005,
                'optimizer_type': 'adamw',
                'scheduler_type': 'cosine'
            },
            {
                'model_type': 'optimized',
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.4,
                'use_batch_norm': True,
                'use_residual': False,
                'learning_rate': 0.001,
                'optimizer_type': 'adam',
                'scheduler_type': 'step'
            }
        ]
        
        # Deep model configs
        deep_configs = [
            {
                'model_type': 'deep',
                'hidden_dims': [256, 128, 64, 32, 16],
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'learning_rate': 0.001,
                'optimizer_type': 'adam',
                'scheduler_type': 'plateau'
            },
            {
                'model_type': 'deep',
                'hidden_dims': [512, 256, 128, 64, 32],
                'dropout_rate': 0.2,
                'use_batch_norm': True,
                'learning_rate': 0.0005,
                'optimizer_type': 'adamw',
                'scheduler_type': 'cosine'
            }
        ]
        
        # Wide model configs
        wide_configs = [
            {
                'model_type': 'wide',
                'hidden_dims': [512, 256],
                'dropout_rate': 0.4,
                'use_batch_norm': True,
                'learning_rate': 0.001,
                'optimizer_type': 'adam',
                'scheduler_type': 'plateau'
            },
            {
                'model_type': 'wide',
                'hidden_dims': [1024, 512],
                'dropout_rate': 0.3,
                'use_batch_norm': True,
                'learning_rate': 0.0005,
                'optimizer_type': 'adamw',
                'scheduler_type': 'step'
            }
        ]
        
        # Combine all configs
        all_configs = optimized_configs + deep_configs + wide_configs
        
        # Add common training parameters
        for config in all_configs:
            config.update({
                'batch_size': 64,
                'num_epochs': 100,
                'early_stopping_patience': 15,
                'weight_decay': 1e-4 if config['optimizer_type'] in ['adamw', 'sgd'] else 0.0
            })
        
        return all_configs[:n_configs]
    
    def evaluate_config(self, config, config_id):
        """
        Evaluate a single hyperparameter configuration.
        
        Args:
            config (dict): Hyperparameter configuration
            config_id (int): Configuration identifier
            
        Returns:
            dict: Evaluation results
        """
        print(f"\n=== Evaluating Configuration {config_id} ===")
        print(f"Model: {config['model_type']}")
        
        try:
            # Set random seeds for reproducibility
            set_random_seeds(self.random_state)
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                self.X_train, self.X_val, self.y_train, self.y_val,
                batch_size=config.get('batch_size', 64)
            )
            
            # Create model based on type
            model_kwargs = {
                'input_dim': self.X_train.shape[1],
                'dropout_rate': config.get('dropout_rate', 0.2)
            }
            
            if config['model_type'] == 'baseline':
                model_kwargs['hidden_dim'] = config.get('hidden_dim', 64)
            elif config['model_type'] in ['optimized', 'deep', 'wide']:
                model_kwargs['hidden_dims'] = config.get('hidden_dims', [128, 64])
                if config['model_type'] == 'optimized':
                    model_kwargs['use_batch_norm'] = config.get('use_batch_norm', True)
                    model_kwargs['use_residual'] = config.get('use_residual', False)
                elif config['model_type'] in ['deep', 'wide']:
                    model_kwargs['use_batch_norm'] = config.get('use_batch_norm', True)
            
            model = create_model(config['model_type'], **model_kwargs)
            
            # Initialize trainer (weight_decay handled by optimizer config)
            trainer = ModelTrainer(
                model=model,
                device=self.device,
                train_loader=train_loader,
                val_loader=val_loader,
                learning_rate=config.get('learning_rate', 0.001),
                optimizer_type=config.get('optimizer_type', 'adam'),
                scheduler_type=config.get('scheduler_type', 'plateau'),
                early_stopping_patience=config.get('early_stopping_patience', 10)
            )
            
            # Apply weight decay to optimizer if specified
            weight_decay = config.get('weight_decay', 0.0)
            if weight_decay > 0.0 and hasattr(trainer.optimizer, 'param_groups'):
                for param_group in trainer.optimizer.param_groups:
                    param_group['weight_decay'] = weight_decay
            
            # Train model
            start_time = datetime.now()
            history = trainer.train(
                num_epochs=config.get('num_epochs', 50),
                print_every=10
            )
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate model
            evaluator = ModelEvaluator(trainer.model, self.device)
            metrics = evaluator.evaluate_model(val_loader)
            
            # Compile results
            result = {
                'config_id': config_id,
                'config': config.copy(),
                'validation_accuracy': metrics['accuracy'],
                'validation_f1': metrics['f1_score'],
                'validation_auc': metrics['roc_auc'],
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'epochs_trained': len(history['train_loss']),
                'training_time': training_time,
                'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'converged': trainer.early_stopping.early_stop if hasattr(trainer, 'early_stopping') else False
            }
            
            print(f"✅ Config {config_id} completed:")
            print(f"   Validation Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1 Score: {metrics['f1_score']:.4f}")
            print(f"   Training Time: {training_time:.1f}s")
            print(f"   Epochs: {result['epochs_trained']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Config {config_id} failed: {str(e)}")
            return {
                'config_id': config_id,
                'config': config.copy(),
                'validation_accuracy': 0.0,
                'error': str(e)
            }
    
    def run_hyperparameter_search(self, n_baseline_configs=20, n_advanced_configs=15):
        """
        Run comprehensive hyperparameter search.
        
        Args:
            n_baseline_configs (int): Number of baseline configurations to test
            n_advanced_configs (int): Number of advanced configurations to test
        """
        print("=== STARTING HYPERPARAMETER SEARCH ===")
        
        # Generate configurations
        print(f"Generating {n_baseline_configs} baseline configurations...")
        baseline_configs = self.generate_baseline_configs(n_baseline_configs)
        
        print(f"Generating {n_advanced_configs} advanced configurations...")
        advanced_configs = self.generate_advanced_configs(n_advanced_configs)
        
        all_configs = baseline_configs + advanced_configs
        total_configs = len(all_configs)
        
        print(f"Total configurations to evaluate: {total_configs}")
        print(f"Estimated time: {total_configs * 2:.0f}-{total_configs * 5:.0f} minutes")
        
        # Evaluate all configurations
        for i, config in enumerate(all_configs, 1):
            print(f"\n{'='*60}")
            print(f"Progress: {i}/{total_configs} ({i/total_configs*100:.1f}%)")
            
            result = self.evaluate_config(config, i)
            self.results.append(result)
            
            # Update best configuration
            if 'validation_accuracy' in result and result['validation_accuracy'] > self.best_score:
                self.best_score = result['validation_accuracy']
                self.best_config = result
                print(f"🏆 New best configuration found! Accuracy: {self.best_score:.4f}")
        
        # Sort results by validation accuracy
        self.results.sort(key=lambda x: x.get('validation_accuracy', 0), reverse=True)
        
        print(f"\n=== HYPERPARAMETER SEARCH COMPLETE ===")
        print(f"Best validation accuracy: {self.best_score:.4f}")
        print(f"Total configurations evaluated: {len(self.results)}")
    
    def analyze_results(self):
        """
        Analyze hyperparameter search results and identify patterns.
        
        Returns:
            dict: Analysis summary
        """
        if not self.results:
            return {}
        
        # Convert results to DataFrame for analysis
        df_results = pd.DataFrame([r for r in self.results if 'validation_accuracy' in r])
        
        if df_results.empty:
            return {}
        
        # Top 5 configurations
        top_5 = df_results.head(5)
        
        # Model type analysis
        model_type_performance = df_results.groupby(
            df_results['config'].apply(lambda x: x['model_type'])
        )['validation_accuracy'].agg(['mean', 'max', 'count']).round(4)
        
        # Optimizer analysis
        optimizer_performance = df_results.groupby(
            df_results['config'].apply(lambda x: x.get('optimizer_type', 'unknown'))
        )['validation_accuracy'].agg(['mean', 'max', 'count']).round(4)
        
        # Learning rate analysis
        lr_performance = df_results.groupby(
            df_results['config'].apply(lambda x: x.get('learning_rate', 0))
        )['validation_accuracy'].agg(['mean', 'max', 'count']).round(4)
        
        # Get columns that exist in the DataFrame
        available_cols = ['config_id', 'validation_accuracy']
        if 'validation_f1' in df_results.columns:
            available_cols.append('validation_f1')
        
        analysis = {
            'total_configs': len(df_results),
            'best_accuracy': df_results['validation_accuracy'].max(),
            'mean_accuracy': df_results['validation_accuracy'].mean(),
            'std_accuracy': df_results['validation_accuracy'].std(),
            'top_5_configs': top_5[available_cols].to_dict('records'),
            'model_type_performance': model_type_performance.to_dict(),
            'optimizer_performance': optimizer_performance.to_dict(),
            'learning_rate_performance': lr_performance.to_dict(),
            'best_config': self.best_config
        }
        
        return analysis
    
    def save_results(self, filepath='experiments/hyperparameter_results.json'):
        """
        Save hyperparameter search results to file.
        
        Args:
            filepath (str): Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Prepare results for JSON serialization
        results_data = {
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_configs': len(self.results),
                'best_accuracy': self.best_score,
                'random_state': self.random_state
            },
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print a comprehensive summary of hyperparameter search results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        analysis = self.analyze_results()
        
        print("\n" + "="*60)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("="*60)
        
        print(f"Total configurations evaluated: {analysis['total_configs']}")
        print(f"Best validation accuracy: {analysis['best_accuracy']:.4f}")
        print(f"Mean validation accuracy: {analysis['mean_accuracy']:.4f} ± {analysis['std_accuracy']:.4f}")
        
        print(f"\n=== TOP 5 CONFIGURATIONS ===")
        for i, config in enumerate(analysis['top_5_configs'], 1):
            print(f"{i}. Config {config['config_id']}: Accuracy={config['validation_accuracy']:.4f}, F1={config['validation_f1']:.4f}")
        
        print(f"\n=== MODEL TYPE PERFORMANCE ===")
        for model_type, stats in analysis['model_type_performance'].items():
            print(f"{model_type}: Mean={stats['mean']:.4f}, Max={stats['max']:.4f}, Count={stats['count']}")
        
        print(f"\n=== BEST CONFIGURATION DETAILS ===")
        if self.best_config:
            best_config = self.best_config['config']
            print(f"Model Type: {best_config['model_type']}")
            print(f"Learning Rate: {best_config.get('learning_rate', 'N/A')}")
            print(f"Optimizer: {best_config.get('optimizer_type', 'N/A')}")
            print(f"Dropout Rate: {best_config.get('dropout_rate', 'N/A')}")
            if 'hidden_dim' in best_config:
                print(f"Hidden Dim: {best_config['hidden_dim']}")
            elif 'hidden_dims' in best_config:
                print(f"Hidden Dims: {best_config['hidden_dims']}")
            print(f"Batch Size: {best_config.get('batch_size', 'N/A')}")
            print(f"Validation Accuracy: {self.best_config['validation_accuracy']:.4f}")


def run_hyperparameter_optimization():
    """
    Main function to run comprehensive hyperparameter optimization.
    """
    print("=== PHASE 5: HYPERPARAMETER OPTIMIZATION ===")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df = load_pickle_data("data/train.pickle")
    if train_df is None:
        print("Failed to load training data")
        return
    
    # Use full dataset for hyperparameter tuning
    print(f"Using full dataset: {len(train_df):,} samples")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(train_df)
    
    # Create train/val splits
    X_train, X_val, y_train, y_val = create_data_splits(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"Train samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(X_train, X_val, y_train, y_val, random_state=42)
    
    # Run hyperparameter search
    tuner.run_hyperparameter_search(n_baseline_configs=25, n_advanced_configs=20)
    
    # Analyze and save results
    tuner.print_summary()
    tuner.save_results()
    
    return tuner


if __name__ == "__main__":
    tuner = run_hyperparameter_optimization() 