"""
Phase 5: Hyperparameter Optimization
Simplified approach to systematically find best model configuration.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

from utils import set_random_seeds, get_device
from data_loader import load_pickle_data
from preprocessor import DataPreprocessor, create_data_splits, create_data_loaders
from model import create_model
from train import ModelTrainer
from evaluate import ModelEvaluator


def run_single_experiment(config, X_train, X_val, y_train, y_val, device, config_id):
    """
    Run a single hyperparameter configuration experiment.
    
    Args:
        config: Hyperparameter configuration dict
        X_train, X_val, y_train, y_val: Data splits
        device: PyTorch device
        config_id: Configuration identifier
        
    Returns:
        dict: Experiment results
    """
    print(f"\n=== Config {config_id}: {config['model_type']} ===")
    print(f"Hidden: {config.get('hidden_dim', config.get('hidden_dims'))}, "
          f"LR: {config['learning_rate']}, Opt: {config['optimizer_type']}")
    
    try:
        # Set random seeds
        set_random_seeds(42)
        
        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_train, X_val, y_train, y_val, batch_size=config['batch_size']
        )
        
        # Create model
        model_kwargs = {'input_dim': X_train.shape[1], 'dropout_rate': config['dropout_rate']}
        
        if config['model_type'] == 'baseline':
            model_kwargs['hidden_dim'] = config['hidden_dim']
        else:
            model_kwargs['hidden_dims'] = config['hidden_dims']
            if config['model_type'] == 'optimized':
                model_kwargs['use_batch_norm'] = config.get('use_batch_norm', True)
                model_kwargs['use_residual'] = config.get('use_residual', False)
            elif config['model_type'] in ['deep', 'wide']:
                model_kwargs['use_batch_norm'] = config.get('use_batch_norm', True)
        
        model = create_model(config['model_type'], **model_kwargs)
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=config['learning_rate'],
            optimizer_type=config['optimizer_type'],
            scheduler_type=config['scheduler_type'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        # Train model
        start_time = datetime.now()
        history = trainer.train(num_epochs=config['num_epochs'], print_every=999)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        evaluator = ModelEvaluator(trainer.model, device)
        metrics = evaluator.evaluate_model(val_loader)
        
        result = {
            'config_id': config_id,
            'model_type': config['model_type'],
            'validation_accuracy': metrics['accuracy'],
            'validation_f1': metrics['f1_score'],
            'validation_auc': metrics['roc_auc'],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'epochs_trained': len(history['train_loss']),
            'training_time': training_time,
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'config': config
        }
        
        print(f"✅ Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}, "
              f"Time: {training_time:.1f}s, Epochs: {result['epochs_trained']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Failed: {str(e)}")
        return {
            'config_id': config_id,
            'validation_accuracy': 0.0,
            'error': str(e),
            'config': config
        }


def generate_baseline_configs():
    """Generate baseline model configurations for optimization."""
    configs = []
    
    # Key baseline configurations to test
    baseline_grid = [
        # Standard configurations
        {'hidden_dim': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_type': 'adam'},
        {'hidden_dim': 128, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_type': 'adam'},
        {'hidden_dim': 256, 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'optimizer_type': 'adam'},
        
        # Different optimizers
        {'hidden_dim': 128, 'dropout_rate': 0.3, 'learning_rate': 0.001, 'optimizer_type': 'adamw'},
        {'hidden_dim': 128, 'dropout_rate': 0.3, 'learning_rate': 0.01, 'optimizer_type': 'sgd'},
        
        # Learning rate variations
        {'hidden_dim': 128, 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'optimizer_type': 'adam'},
        {'hidden_dim': 128, 'dropout_rate': 0.3, 'learning_rate': 0.005, 'optimizer_type': 'adam'},
        
        # Dropout variations
        {'hidden_dim': 128, 'dropout_rate': 0.1, 'learning_rate': 0.001, 'optimizer_type': 'adam'},
        {'hidden_dim': 128, 'dropout_rate': 0.5, 'learning_rate': 0.001, 'optimizer_type': 'adam'},
        
        # Size variations
        {'hidden_dim': 32, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'optimizer_type': 'adam'},
        {'hidden_dim': 512, 'dropout_rate': 0.4, 'learning_rate': 0.0005, 'optimizer_type': 'adam'},
    ]
    
    for config in baseline_grid:
        config.update({
            'model_type': 'baseline',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 100,
            'early_stopping_patience': 15
        })
        configs.append(config)
    
    return configs


def generate_advanced_configs():
    """Generate advanced model configurations."""
    configs = []
    
    # Optimized models
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
    
    # Deep models
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
    
    # Wide models
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
    
    all_configs = optimized_configs + deep_configs + wide_configs
    
    # Add common parameters
    for config in all_configs:
        config.update({
            'batch_size': 64,
            'num_epochs': 100,
            'early_stopping_patience': 15
        })
    
    return all_configs


def run_phase5_optimization():
    """Main function for Phase 5 optimization."""
    print("=== PHASE 5: HYPERPARAMETER OPTIMIZATION ===\n")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df = load_pickle_data("data/train.pickle")
    if train_df is None:
        print("Failed to load training data")
        return
    
    print(f"Loaded {len(train_df):,} training samples")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(train_df)
    
    # Create splits
    X_train, X_val, y_train, y_val = create_data_splits(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train):,}, Validation: {len(X_val):,}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate configurations
    print("\nGenerating configurations...")
    baseline_configs = generate_baseline_configs()
    advanced_configs = generate_advanced_configs()
    
    all_configs = baseline_configs + advanced_configs
    total_configs = len(all_configs)
    
    print(f"Baseline configs: {len(baseline_configs)}")
    print(f"Advanced configs: {len(advanced_configs)}")
    print(f"Total configs: {total_configs}")
    print(f"Estimated time: {total_configs * 3:.0f}-{total_configs * 8:.0f} minutes\n")
    
    # Run experiments
    results = []
    best_accuracy = 0.0
    best_config = None
    
    for i, config in enumerate(all_configs, 1):
        print(f"Progress: {i}/{total_configs} ({i/total_configs*100:.1f}%)")
        
        result = run_single_experiment(config, X_train, X_val, y_train, y_val, device, i)
        results.append(result)
        
        # Track best result
        if 'validation_accuracy' in result and result['validation_accuracy'] > best_accuracy:
            best_accuracy = result['validation_accuracy']
            best_config = result
            print(f"🏆 NEW BEST: {best_accuracy:.4f}")
    
    # Sort results
    results.sort(key=lambda x: x.get('validation_accuracy', 0), reverse=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Configurations evaluated: {len(results)}")
    
    # Top 5 results
    print(f"\nTOP 5 CONFIGURATIONS:")
    for i, result in enumerate(results[:5], 1):
        if 'validation_accuracy' in result:
            config = result['config']
            print(f"{i}. Accuracy: {result['validation_accuracy']:.4f} | "
                  f"Model: {result['model_type']} | "
                  f"LR: {config['learning_rate']} | "
                  f"Opt: {config['optimizer_type']} | "
                  f"Epochs: {result.get('epochs_trained', 'N/A')}")
    
    # Best configuration details
    if best_config:
        print(f"\nBEST CONFIGURATION DETAILS:")
        config = best_config['config']
        print(f"Model Type: {config['model_type']}")
        print(f"Architecture: {config.get('hidden_dim', config.get('hidden_dims'))}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Optimizer: {config['optimizer_type']}")
        print(f"Dropout: {config['dropout_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Validation Accuracy: {best_config['validation_accuracy']:.4f}")
        print(f"Training Time: {best_config.get('training_time', 'N/A'):.1f}s")
        print(f"Parameters: {best_config.get('parameters', 'N/A'):,}")
    
    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'best_accuracy': best_accuracy,
        'total_configs': len(results),
        'results': results,
        'best_config': best_config
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/phase5_optimization_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to experiments/phase5_optimization_results.json")
    
    return results, best_config


if __name__ == "__main__":
    results, best_config = run_phase5_optimization() 