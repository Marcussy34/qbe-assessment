"""
Final Phase 5 Demo: Working Hyperparameter Optimization
Demonstrates systematic optimization with proper error handling.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
from datetime import datetime
import json
import os

from utils import set_random_seeds, get_device
from data_loader import load_pickle_data
from preprocessor import DataPreprocessor, create_data_splits, create_data_loaders
from model import create_model
from train import ModelTrainer


def calculate_metrics(model, data_loader, device):
    """Calculate evaluation metrics manually."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = (output > 0.5).float()
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = correct / total
    
    # Simple F1 calculation
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    targets_binary = np.array(all_targets).astype(int)
    
    tp = np.sum((predictions_binary == 1) & (targets_binary == 1))
    fp = np.sum((predictions_binary == 1) & (targets_binary == 0))
    fn = np.sum((predictions_binary == 0) & (targets_binary == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall
    }


def run_single_config(config, X_train, X_val, y_train, y_val, device, config_id):
    """Run a single configuration experiment."""
    print(f"\n=== Config {config_id}: {config['model_type']} ===")
    
    try:
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
        
        # Evaluate manually
        metrics = calculate_metrics(trainer.model, val_loader, device)
        
        result = {
            'config_id': config_id,
            'model_type': config['model_type'],
            'validation_accuracy': metrics['accuracy'],
            'validation_f1': metrics['f1_score'],
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
            'model_type': config.get('model_type', 'unknown'),
            'validation_accuracy': 0.0,
            'error': str(e),
            'config': config
        }


def generate_test_configs():
    """Generate key configurations to test."""
    return [
        # Best baseline configs
        {
            'model_type': 'baseline',
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'optimizer_type': 'adam',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 20,
            'early_stopping_patience': 5
        },
        {
            'model_type': 'baseline',
            'hidden_dim': 256,
            'dropout_rate': 0.2,
            'learning_rate': 0.0005,
            'optimizer_type': 'adam',
            'scheduler_type': 'plateau',
            'batch_size': 64,
            'num_epochs': 20,
            'early_stopping_patience': 5
        },
        {
            'model_type': 'baseline',
            'hidden_dim': 128,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'optimizer_type': 'adamw',
            'scheduler_type': 'cosine',
            'batch_size': 64,
            'num_epochs': 20,
            'early_stopping_patience': 5
        },
        # Advanced models
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
            'num_epochs': 20,
            'early_stopping_patience': 5
        }
    ]


def run_final_phase5_demo():
    """Run the final Phase 5 demonstration."""
    print("=== FINAL PHASE 5 DEMO: HYPERPARAMETER OPTIMIZATION ===\n")
    
    # Load data
    print("Loading and preprocessing data...")
    train_df = load_pickle_data("data/train.pickle")
    if train_df is None:
        print("Failed to load training data")
        return
    
    # Use subset for demo
    subset_size = 30000
    train_subset = train_df.sample(n=subset_size, random_state=42)
    print(f"Using subset: {len(train_subset):,} samples")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(train_subset)
    
    # Create splits
    X_train, X_val, y_train, y_val = create_data_splits(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {len(X_train):,}, Validation: {len(X_val):,}")
    
    device = get_device()
    print(f"Device: {device}")
    
    # Run experiments
    configs = generate_test_configs()
    print(f"\nTesting {len(configs)} configurations...\n")
    
    results = []
    best_accuracy = 0.0
    best_config = None
    
    for i, config in enumerate(configs, 1):
        print(f"Progress: {i}/{len(configs)}")
        
        result = run_single_config(config, X_train, X_val, y_train, y_val, device, i)
        results.append(result)
        
        if 'validation_accuracy' in result and result['validation_accuracy'] > best_accuracy:
            best_accuracy = result['validation_accuracy']
            best_config = result
            print(f"🏆 NEW BEST: {best_accuracy:.4f}")
    
    # Sort results
    results.sort(key=lambda x: x.get('validation_accuracy', 0), reverse=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("PHASE 5 OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Dataset size: {subset_size:,} samples")
    
    print(f"\nRANKED RESULTS:")
    for i, result in enumerate(results, 1):
        model_type = result.get('model_type', 'unknown')
        if 'validation_accuracy' in result:
            config = result['config']
            arch = config.get('hidden_dim', config.get('hidden_dims'))
            print(f"{i}. {result['validation_accuracy']:.4f} | {model_type:>10} | "
                  f"arch={arch} | lr={config['learning_rate']} | "
                  f"opt={config['optimizer_type']} | epochs={result.get('epochs_trained', 'N/A')}")
        else:
            error = result.get('error', 'Unknown error')[:50]
            print(f"{i}. FAILED | {model_type} | {error}")
    
    # Best configuration analysis
    if best_config:
        print(f"\n🏆 BEST CONFIGURATION:")
        config = best_config['config']
        print(f"Model Type: {config['model_type']}")
        print(f"Architecture: {config.get('hidden_dim', config.get('hidden_dims'))}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Optimizer: {config['optimizer_type']}")
        print(f"Scheduler: {config['scheduler_type']}")
        print(f"Dropout: {config['dropout_rate']}")
        print(f"Validation Accuracy: {best_config['validation_accuracy']:.4f}")
        print(f"F1 Score: {best_config.get('validation_f1', 0):.4f}")
        print(f"Training Time: {best_config.get('training_time', 0):.1f}s")
        print(f"Parameters: {best_config.get('parameters', 0):,}")
        
        # Performance improvement
        baseline_acc = 0.5115
        improvement = best_config['validation_accuracy'] - baseline_acc
        print(f"\n📈 IMPROVEMENT OVER BASELINE:")
        print(f"Baseline: {baseline_acc:.4f}")
        print(f"Best: {best_config['validation_accuracy']:.4f}")
        print(f"Improvement: +{improvement:.4f} ({improvement/baseline_acc*100:.1f}%)")
    
    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'subset_size': subset_size,
        'best_accuracy': best_accuracy,
        'results': results,
        'best_config': best_config
    }
    
    os.makedirs('experiments', exist_ok=True)
    with open('experiments/final_phase5_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\nResults saved to experiments/final_phase5_results.json")
    
    # Final recommendations
    print(f"\n🚀 PHASE 5 CONCLUSIONS:")
    print(f"1. Best performing model: {best_config['config']['model_type']}")
    print(f"2. Optimal accuracy achieved: {best_accuracy:.4f}")
    print(f"3. Ready for full dataset training")
    print(f"4. Architecture scaling shows promise")
    
    return results, best_config


if __name__ == "__main__":
    results, best_config = run_final_phase5_demo() 