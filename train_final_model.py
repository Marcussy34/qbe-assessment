"""
Phase 6: Full Dataset Training with Optimal Configuration
Train the best model from Phase 5 on the complete 439K dataset.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

from utils import set_random_seeds, get_device, print_system_info
from data_loader import load_pickle_data
from preprocessor import DataPreprocessor, create_data_splits, create_data_loaders
from model import create_model, model_summary
from train import ModelTrainer
from evaluate import ModelEvaluator


def calculate_metrics(model, data_loader, device):
    """Calculate comprehensive evaluation metrics."""
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
    
    # Calculate comprehensive metrics
    predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
    targets_binary = np.array(all_targets).astype(int)
    
    tp = np.sum((predictions_binary == 1) & (targets_binary == 1))
    fp = np.sum((predictions_binary == 1) & (targets_binary == 0))
    fn = np.sum((predictions_binary == 0) & (targets_binary == 1))
    tn = np.sum((predictions_binary == 0) & (targets_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC AUC approximation
    from sklearn.metrics import roc_auc_score
    try:
        roc_auc = roc_auc_score(targets_binary, all_predictions)
    except:
        roc_auc = 0.5
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
    }


def get_optimal_config():
    """Return the best configuration from Phase 5."""
    return {
        'model_type': 'baseline',
        'hidden_dim': 256,
        'dropout_rate': 0.2,
        'learning_rate': 0.0005,
        'optimizer_type': 'adam',
        'scheduler_type': 'plateau',
        'batch_size': 64,
        'num_epochs': 100,  # More epochs for full dataset
        'early_stopping_patience': 15
    }


def train_final_model():
    """Train the final optimized model on the complete dataset."""
    print("=== PHASE 6: FINAL MODEL TRAINING ON FULL DATASET ===\n")
    
    # System info
    print_system_info()
    device = get_device()
    set_random_seeds(42)
    
    # Load full dataset
    print("Loading complete training dataset...")
    train_df = load_pickle_data("data/train.pickle")
    if train_df is None:
        print("❌ Failed to load training data")
        return None
    
    print(f"✅ Loaded complete dataset: {len(train_df):,} samples")
    print(f"Memory usage: ~{len(train_df) * 14 * 4 / 1024**2:.1f} MB (estimated)")
    
    # Preprocess data
    print("\nPreprocessing complete dataset...")
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(train_df)
    
    # Create stratified train/val splits
    print("Creating stratified train/validation splits...")
    X_train, X_val, y_train, y_val = create_data_splits(
        X_processed, y, test_size=0.15, random_state=42  # Smaller val set for more training data
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Train/Val split: {len(X_train)/len(X_processed)*100:.1f}% / {len(X_val)/len(X_processed)*100:.1f}%")
    
    # Get optimal configuration
    config = get_optimal_config()
    print(f"\n🏆 Using optimal configuration from Phase 5:")
    print(f"Model: {config['model_type']} with {config['hidden_dim']} hidden units")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Optimizer: {config['optimizer_type']}")
    print(f"Scheduler: {config['scheduler_type']}")
    print(f"Dropout: {config['dropout_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max epochs: {config['num_epochs']}")
    
    # Create data loaders
    print(f"\nCreating data loaders with batch size {config['batch_size']}...")
    train_loader, val_loader = create_data_loaders(
        X_train, X_val, y_train, y_val, batch_size=config['batch_size']
    )
    
    print(f"Training batches: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    
    # Create model
    print(f"\nCreating {config['model_type']} model...")
    model = create_model(
        config['model_type'],
        input_dim=X_train.shape[1],
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate']
    )
    
    # Display model summary
    model_summary(model, input_dim=X_train.shape[1])
    
    # Create trainer
    print(f"\nInitializing trainer...")
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
    print(f"\n🚀 Starting training on full dataset...")
    print(f"Target: Beat 57.97% validation accuracy from Phase 5")
    print(f"Estimated training time: 15-30 minutes")
    print("-" * 60)
    
    start_time = datetime.now()
    history = trainer.train(
        num_epochs=config['num_epochs'],
        print_every=5  # Print every 5 epochs for full dataset
    )
    training_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"Final training accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    
    # Comprehensive evaluation
    print(f"\n📊 Evaluating final model performance...")
    final_metrics = calculate_metrics(trainer.model, val_loader, device)
    
    print(f"\n🎯 FINAL MODEL PERFORMANCE:")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"F1 Score: {final_metrics['f1_score']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"Specificity: {final_metrics['specificity']:.4f}")
    print(f"ROC AUC: {final_metrics['roc_auc']:.4f}")
    
    # Confusion matrix
    cm = final_metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"True Positives:  {cm['tp']:,}")
    print(f"False Positives: {cm['fp']:,}")
    print(f"True Negatives:  {cm['tn']:,}")
    print(f"False Negatives: {cm['fn']:,}")
    
    # Performance comparison
    phase5_accuracy = 0.5797  # Best from Phase 5
    improvement = final_metrics['accuracy'] - phase5_accuracy
    print(f"\n📈 PERFORMANCE COMPARISON:")
    print(f"Phase 5 (30K subset): {phase5_accuracy:.4f}")
    print(f"Phase 6 (full dataset): {final_metrics['accuracy']:.4f}")
    print(f"Improvement: {improvement:+.4f} ({improvement/phase5_accuracy*100:+.1f}%)")
    
    # Save model
    model_path = "models/final_optimized_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(trainer.model.state_dict(), model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save preprocessor for test evaluation
    import pickle
    preprocessor_path = "models/final_preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Save complete results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(train_df),
        'config': config,
        'final_metrics': final_metrics,
        'training_history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'train_acc': history['train_acc'],
            'val_acc': history['val_acc'],
            'learning_rates': history.get('learning_rates', [])
        },
        'training_time_minutes': training_time / 60,
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'improvement_over_phase5': improvement
    }
    
    results_path = "experiments/phase6_final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Complete results saved to {results_path}")
    
    # Performance analysis
    print(f"\n🔍 PERFORMANCE ANALYSIS:")
    if final_metrics['accuracy'] > 0.60:
        print(f"🎉 EXCELLENT: >60% accuracy achieved!")
    elif final_metrics['accuracy'] > phase5_accuracy:
        print(f"✅ SUCCESS: Improved over Phase 5 baseline")
    else:
        print(f"⚠️  WARNING: No improvement over subset training")
    
    if final_metrics['f1_score'] > 0.65:
        print(f"🎯 HIGH F1: Balanced precision and recall")
    
    # Next steps recommendation
    print(f"\n🚀 NEXT STEPS RECOMMENDATIONS:")
    if final_metrics['accuracy'] < 0.95:
        print(f"1. Ensemble multiple models for higher accuracy")
        print(f"2. Advanced regularization techniques")
        print(f"3. Feature selection and engineering")
        print(f"4. Alternative model architectures")
    else:
        print(f"🏆 TARGET ACHIEVED! Ready for test evaluation")
    
    return {
        'model': trainer.model,
        'preprocessor': preprocessor,
        'metrics': final_metrics,
        'config': config,
        'history': history
    }


if __name__ == "__main__":
    result = train_final_model() 