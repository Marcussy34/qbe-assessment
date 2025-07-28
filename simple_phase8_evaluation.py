#!/usr/bin/env python3
"""
Phase 8 Simple Model Evaluation
Compare neural network models and create ensemble predictions
"""

import sys
import numpy as np
import pandas as pd
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_loader import load_pickle_data
from model import BaselineNet, OptimizedNet

def load_model_with_architecture(model_path):
    """
    Load a model by trying different architectures
    """
    try:
        # Try different common architectures
        architectures = [
            (BaselineNet, {'input_dim': 14, 'hidden_dim': 64}),
            (BaselineNet, {'input_dim': 14, 'hidden_dim': 128}),
            (BaselineNet, {'input_dim': 14, 'hidden_dim': 256}),
            (BaselineNet, {'input_dim': 14, 'hidden_dim': 512}),
            (OptimizedNet, {'input_dim': 14, 'hidden_dims': [256, 128], 'dropout_rate': 0.2}),
            (OptimizedNet, {'input_dim': 14, 'hidden_dims': [256, 128, 64], 'dropout_rate': 0.3}),
        ]
        
        state_dict = torch.load(model_path, map_location='cpu')
        
        for model_class, params in architectures:
            try:
                model = model_class(**params)
                model.load_state_dict(state_dict)
                print(f"✅ Loaded {model_path} with {model_class.__name__}({params})")
                return model, f"{model_class.__name__}({params})"
            except:
                continue
                
        print(f"❌ Could not load {model_path} with any architecture")
        return None, None
        
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None, None

def evaluate_model(model, X_val, y_val, model_name):
    """
    Evaluate a single model
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val)
        predictions = model(X_tensor).squeeze().numpy()
        binary_predictions = (predictions > 0.5).astype(int)
        
        accuracy = (binary_predictions == y_val).mean()
        
        # Calculate additional metrics
        tp = ((binary_predictions == 1) & (y_val == 1)).sum()
        tn = ((binary_predictions == 0) & (y_val == 0)).sum()
        fp = ((binary_predictions == 1) & (y_val == 0)).sum()
        fn = ((binary_predictions == 0) & (y_val == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'predictions': predictions.tolist()
        }

def simple_ensemble_predict(models, X_val):
    """
    Simple ensemble - average predictions
    """
    all_predictions = []
    
    for model, _ in models:
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_val)
            predictions = model(X_tensor).squeeze().numpy()
            all_predictions.append(predictions)
    
    # Average predictions
    ensemble_predictions = np.mean(all_predictions, axis=0)
    return ensemble_predictions

def main():
    print("🚀 Phase 8: Simple Model Evaluation & Comparison")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # Load data
    print("📊 Loading data...")
    train_data = load_pickle_data('data/train.pickle')
    test_data = load_pickle_data('data/test.pickle')
    
    # Load preprocessor
    with open('models/final_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Prepare validation data (use a subset for quick evaluation)
    X_full = preprocessor.transform(train_data.drop('is_correct', axis=1))
    y_full = train_data['is_correct'].values.astype(int)
    
    # Use last 20% as validation set
    val_size = int(0.2 * len(X_full))
    X_val = X_full[-val_size:]
    y_val = y_full[-val_size:]
    
    print(f"✅ Validation set: {len(X_val):,} samples")
    
    # Find all model files
    model_files = [
        'models/final_optimized_model.pth',
        'models/test_baseline_model.pth',
        'models/ensemble_baseline_128.pth',
        'models/ensemble_optimized_high_dropout.pth',
        'models/ensemble_wide_512.pth'
    ]
    
    # Load and evaluate models
    print("\n🔍 Loading and evaluating models...")
    loaded_models = []
    all_results = []
    
    for model_file in model_files:
        if Path(model_file).exists():
            model, arch_info = load_model_with_architecture(model_file)
            if model is not None:
                loaded_models.append((model, arch_info))
                result = evaluate_model(model, X_val, y_val, f"{Path(model_file).stem} ({arch_info})")
                all_results.append(result)
                print(f"📊 {result['model_name']}: {result['accuracy']:.4f} accuracy")
        else:
            print(f"⚠️  Model not found: {model_file}")
    
    print(f"\n✅ Evaluated {len(loaded_models)} models")
    
    # Create ensemble predictions if we have multiple models
    if len(loaded_models) >= 2:
        print("\n🔧 Creating ensemble predictions...")
        ensemble_predictions = simple_ensemble_predict(loaded_models, X_val)
        binary_ensemble = (ensemble_predictions > 0.5).astype(int)
        ensemble_accuracy = (binary_ensemble == y_val).mean()
        
        # Calculate ensemble metrics
        tp = ((binary_ensemble == 1) & (y_val == 1)).sum()
        tn = ((binary_ensemble == 0) & (y_val == 0)).sum()
        fp = ((binary_ensemble == 1) & (y_val == 0)).sum()
        fn = ((binary_ensemble == 0) & (y_val == 1)).sum()
        
        ensemble_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        ensemble_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        ensemble_f1 = 2 * (ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall) if (ensemble_precision + ensemble_recall) > 0 else 0
        
        ensemble_result = {
            'model_name': f'Simple Ensemble ({len(loaded_models)} models)',
            'accuracy': float(ensemble_accuracy),
            'precision': float(ensemble_precision),
            'recall': float(ensemble_recall),
            'f1_score': float(ensemble_f1),
            'predictions': ensemble_predictions.tolist()
        }
        all_results.append(ensemble_result)
        print(f"📊 {ensemble_result['model_name']}: {ensemble_result['accuracy']:.4f} accuracy")
    
    # Find best model
    best_result = max(all_results, key=lambda x: x['accuracy'])
    baseline_accuracy = 0.6424  # Phase 6 baseline
    
    print(f"\n🏆 Best Model: {best_result['model_name']}")
    print(f"🎯 Best Accuracy: {best_result['accuracy']:.4f}")
    print(f"📈 vs Phase 6 Baseline: {best_result['accuracy'] - baseline_accuracy:+.4f}")
    
    # Generate test predictions with best model
    print(f"\n🧪 Generating test predictions...")
    X_test = preprocessor.transform(test_data)
    
    if 'Ensemble' in best_result['model_name']:
        test_predictions = simple_ensemble_predict(loaded_models, X_test)
    else:
        # Find the best individual model
        best_model = None
        for i, result in enumerate(all_results[:-1]):  # Exclude ensemble from individual models
            if result['model_name'] == best_result['model_name']:
                best_model = loaded_models[i][0]
                break
        
        if best_model:
            best_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                test_predictions = best_model(X_test_tensor).squeeze().numpy()
    
    # Save results
    phase8_results = {
        'timestamp': start_time.isoformat(),
        'evaluation_set_size': len(X_val),
        'models_evaluated': len(loaded_models),
        'individual_results': all_results[:-1] if len(loaded_models) >= 2 else all_results,
        'ensemble_result': all_results[-1] if len(loaded_models) >= 2 and 'Ensemble' in all_results[-1]['model_name'] else None,
        'best_model': best_result['model_name'],
        'best_accuracy': best_result['accuracy'],
        'improvement_vs_baseline': best_result['accuracy'] - baseline_accuracy,
        'baseline_accuracy': baseline_accuracy,
        'test_predictions_count': len(test_predictions),
        'test_prediction_stats': {
            'mean': float(np.mean(test_predictions)),
            'std': float(np.std(test_predictions)),
            'min': float(np.min(test_predictions)),
            'max': float(np.max(test_predictions)),
            'positive_predictions': int(np.sum(test_predictions > 0.5)),
            'negative_predictions': int(np.sum(test_predictions <= 0.5))
        }
    }
    
    # Save results
    with open('experiments/phase8_evaluation_results.json', 'w') as f:
        json.dump(phase8_results, f, indent=2)
    
    # Save test predictions
    test_df = pd.DataFrame({
        'prediction': test_predictions,
        'binary_prediction': (test_predictions > 0.5).astype(int)
    })
    test_df.to_csv('experiments/phase8_test_predictions.csv', index=False)
    
    # Print summary table
    print(f"\n📊 Phase 8 Model Comparison Summary:")
    print("-" * 80)
    print(f"{'Model':<50} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    for result in all_results:
        print(f"{result['model_name']:<50} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f}")
    print("-" * 80)
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Phase 8 evaluation completed in {elapsed_time:.1f}s")
    print(f"💾 Results saved to: experiments/phase8_evaluation_results.json")
    print(f"💾 Test predictions saved to: experiments/phase8_test_predictions.csv")
    
    return phase8_results

if __name__ == "__main__":
    results = main() 