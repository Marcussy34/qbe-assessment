"""
Quick Phase 5 Demo: Hyperparameter Optimization
Fast demonstration of systematic optimization approach on subset data.
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


def run_quick_experiment(config, X_train, X_val, y_train, y_val, device, config_id):
    """
    Run a single experiment quickly on subset data.
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
        
        # Train model (reduced epochs for speed)
        start_time = datetime.now()
        history = trainer.train(num_epochs=config['num_epochs'], print_every=999)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        evaluator = ModelEvaluator(trainer.model, device)
        metrics = evaluator.evaluate_data_loader(val_loader)
        
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


def generate_key_configs():
    """Generate the most promising configurations for quick testing."""
    configs = [
        # Baseline variations - focusing on most promising
        {
            'model_type': 'baseline',
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 30,
            'early_stopping_patience': 8
        },
        {
            'model_type': 'baseline',
            'hidden_dim': 256,
            'dropout_rate': 0.2,
            'learning_rate': 0.0005,
            'optimizer_type': 'adam',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 30,
            'early_stopping_patience': 8
        },
        {
            'model_type': 'baseline',
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'optimizer_type': 'adamw',
            'scheduler_type': 'cosine',
            'batch_size': 64,
            'num_epochs': 30,
            'early_stopping_patience': 8
        },
        
        # Optimized model - best configuration
        {
            'model_type': 'optimized',
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.2,
            'use_batch_norm': True,
            'use_residual': True,
            'learning_rate': 0.0005,
            'optimizer_type': 'adamw',
            'scheduler_type': 'cosine',
            'batch_size': 64,
            'num_epochs': 30,
            'early_stopping_patience': 8
        },
        
        # Deep model
        {
            'model_type': 'deep',
            'hidden_dims': [256, 128, 64, 32, 16],
            'dropout_rate': 0.3,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 30,
            'early_stopping_patience': 8
        },
        
        # Wide model
        {
            'model_type': 'wide',
            'hidden_dims': [512, 256],
            'dropout_rate': 0.4,
            'use_batch_norm': True,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 30,
            'early_stopping_patience': 8
        }
    ]
    
    return configs


def run_quick_phase5_demo():
    """Quick demonstration of Phase 5 optimization approach."""
    print("=== QUICK PHASE 5 DEMO: HYPERPARAMETER OPTIMIZATION ===\n")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df = load_pickle_data("data/train.pickle")
    if train_df is None:
        print("Failed to load training data")
        return
    
    # Use subset for quick demo (50k samples)
    subset_size = 50000
    train_subset = train_df.sample(n=subset_size, random_state=42)
    print(f"Using subset: {len(train_subset):,} samples (from {len(train_df):,} total)")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(train_subset)
    
    # Create splits
    X_train, X_val, y_train, y_val = create_data_splits(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train):,}, Validation: {len(X_val):,}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate key configurations
    print("\nGenerating key configurations for testing...")
    configs = generate_key_configs()
    total_configs = len(configs)
    
    print(f"Testing {total_configs} key configurations")
    print(f"Estimated time: {total_configs * 2:.0f}-{total_configs * 4:.0f} minutes\n")
    
    # Run experiments
    results = []
    best_accuracy = 0.0
    best_config = None
    
    for i, config in enumerate(configs, 1):
        print(f"Progress: {i}/{total_configs} ({i/total_configs*100:.1f}%)")
        
        result = run_quick_experiment(config, X_train, X_val, y_train, y_val, device, i)
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
    print("QUICK OPTIMIZATION DEMO COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Configurations tested: {len(results)}")
    print(f"Dataset size: {subset_size:,} samples")
    
    # Show all results
    print(f"\nALL CONFIGURATIONS RANKED:")
    for i, result in enumerate(results, 1):
        if 'validation_accuracy' in result:
            config = result['config']
            arch = config.get('hidden_dim', config.get('hidden_dims'))
            model_type = result.get('model_type', config.get('model_type', 'unknown'))
            print(f"{i}. {result['validation_accuracy']:.4f} | "
                  f"{model_type:>10} | "
                  f"arch={arch} | "
                  f"lr={config['learning_rate']} | "
                  f"opt={config['optimizer_type']} | "
                  f"epochs={result.get('epochs_trained', 'N/A')}")
        else:
            config = result.get('config', {})
            model_type = config.get('model_type', 'unknown')
            print(f"{i}. FAILED | {model_type} | {result.get('error', 'Unknown error')}")
    
    # Best configuration details
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION ANALYSIS:")
        config = best_config['config']
        print(f"Model Type: {config['model_type']}")
        print(f"Architecture: {config.get('hidden_dim', config.get('hidden_dims'))}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Optimizer: {config['optimizer_type']}")
        print(f"Scheduler: {config['scheduler_type']}")
        print(f"Dropout Rate: {config['dropout_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Validation Accuracy: {best_config['validation_accuracy']:.4f}")
        print(f"Validation F1: {best_config.get('validation_f1', 'N/A'):.4f}")
        print(f"Training Time: {best_config.get('training_time', 0):.1f}s")
        print(f"Parameters: {best_config.get('parameters', 0):,}")
        print(f"Epochs Trained: {best_config.get('epochs_trained', 'N/A')}")
        
        # Performance improvement analysis
        baseline_acc = 0.5115  # Random baseline from previous phases
        improvement = best_config['validation_accuracy'] - baseline_acc
        print(f"\n📈 PERFORMANCE IMPROVEMENT:")
        print(f"Baseline accuracy: {baseline_acc:.4f}")
        print(f"Best accuracy: {best_config['validation_accuracy']:.4f}")
        print(f"Improvement: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
        
        # Extrapolation to full dataset
        print(f"\n🎯 EXTRAPOLATION TO FULL DATASET:")
        print(f"Current subset size: {subset_size:,}")
        print(f"Full dataset size: {len(train_df):,}")
        print(f"Scale factor: {len(train_df)/subset_size:.1f}x")
        print(f"Expected training time on full dataset: {best_config.get('training_time', 0) * (len(train_df)/subset_size):.0f}s")
    
    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'subset_size': subset_size,
        'full_dataset_size': len(train_df),
        'best_accuracy': best_accuracy,
        'total_configs': len(results),
        'results': results,
        'best_config': best_config
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/quick_phase5_demo_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to experiments/quick_phase5_demo_results.json")
    
    # Recommendations for full Phase 5
    print(f"\n🚀 RECOMMENDATIONS FOR FULL PHASE 5:")
    print(f"1. Best model type identified: {best_config['config']['model_type']}")
    print(f"2. Optimal learning rate range: 0.0005-0.001")
    print(f"3. Preferred optimizer: {best_config['config']['optimizer_type']}")
    print(f"4. Run full optimization on complete dataset")
    print(f"5. Focus on top 3 architectures with longer training")
    
    return results, best_config


if __name__ == "__main__":
    results, best_config = run_quick_phase5_demo() 