#!/usr/bin/env python3
"""
Phase 9: Proper Validation Methodology
Re-evaluate all models using proper stratified validation splits to correct Phase 8 data leakage
"""

import sys
import numpy as np
import pandas as pd
import torch
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from data_loader import load_pickle_data
from model import BaselineNet, OptimizedNet

def load_model_with_architecture(model_path):
    """Load a model by trying different architectures"""
    try:
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
    """Evaluate a single model with comprehensive metrics"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val)
        predictions = model(X_tensor).squeeze().numpy()
        binary_predictions = (predictions > 0.5).astype(int)
        
        accuracy = (binary_predictions == y_val).mean()
        
        # Calculate detailed metrics
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
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'predictions': predictions.tolist()
        }

def simple_ensemble_predict(models, X_val):
    """Simple ensemble - average predictions from multiple models"""
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
    print("🔍 Phase 9: Proper Validation Methodology")
    print("=" * 60)
    print("⚠️  Correcting Phase 8 data leakage by using proper stratified validation")
    
    start_time = datetime.now()
    
    # Load data
    print("📊 Loading data...")
    train_data = load_pickle_data('data/train.pickle')
    test_data = load_pickle_data('data/test.pickle')
    
    # Load preprocessor
    print("🔧 Loading preprocessor...")
    with open('models/final_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Prepare data with PROPER stratified split (same as Phase 6)
    print("✂️  Creating proper stratified validation split (85/15)...")
    X_full = preprocessor.transform(train_data.drop('is_correct', axis=1))
    y_full = train_data['is_correct'].values.astype(int)
    
    # Use the SAME stratified split approach as Phase 6
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, 
        test_size=0.15,  # Same as Phase 6
        random_state=42,  # Same seed for reproducibility
        stratify=y_full   # Stratified split to maintain class balance
    )
    
    print(f"✅ Training set: {len(X_train):,} samples")
    print(f"✅ Validation set: {len(X_val):,} samples")
    print(f"📊 Train positive rate: {y_train.mean():.4f}")
    print(f"📊 Val positive rate: {y_val.mean():.4f}")
    
    # Find all model files
    model_files = [
        'models/final_optimized_model.pth',
        'models/test_baseline_model.pth',
        'models/ensemble_baseline_128.pth',
        'models/ensemble_optimized_high_dropout.pth',
        'models/ensemble_wide_512.pth'
    ]
    
    # Load and evaluate models with PROPER validation
    print("\n🔍 Loading and evaluating models with proper validation...")
    loaded_models = []
    all_results = []
    
    for model_file in model_files:
        if Path(model_file).exists():
            model, arch_info = load_model_with_architecture(model_file)
            if model is not None:
                loaded_models.append((model, arch_info))
                result = evaluate_model(model, X_val, y_val, f"{Path(model_file).stem}")
                result['architecture'] = arch_info
                all_results.append(result)
                print(f"📊 {result['model_name']}: {result['accuracy']:.4f} accuracy (F1: {result['f1_score']:.4f})")
        else:
            print(f"⚠️  Model not found: {model_file}")
    
    print(f"\n✅ Evaluated {len(loaded_models)} models with proper validation")
    
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
            'model_name': f'Ensemble_{len(loaded_models)}_models',
            'architecture': f'Simple Average of {len(loaded_models)} models',
            'accuracy': float(ensemble_accuracy),
            'precision': float(ensemble_precision),
            'recall': float(ensemble_recall),
            'f1_score': float(ensemble_f1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'predictions': ensemble_predictions.tolist()
        }
        all_results.append(ensemble_result)
        print(f"📊 {ensemble_result['model_name']}: {ensemble_result['accuracy']:.4f} accuracy (F1: {ensemble_result['f1_score']:.4f})")
    
    # Compare with Phase 6 and Phase 8 baselines
    phase6_baseline = 0.6424  # Phase 6 proper validation
    phase8_claimed = 0.7994   # Phase 8 problematic validation
    
    # Find best model
    best_result = max(all_results, key=lambda x: x['accuracy'])
    
    print(f"\n🏆 PROPER VALIDATION RESULTS:")
    print("=" * 60)
    print(f"🥇 Best Model: {best_result['model_name']}")
    print(f"🎯 Best Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"📊 Best F1 Score: {best_result['f1_score']:.4f}")
    print()
    print(f"📈 vs Phase 6 Baseline: {best_result['accuracy'] - phase6_baseline:+.4f} ({((best_result['accuracy'] - phase6_baseline)/phase6_baseline)*100:+.2f}%)")
    print(f"❌ vs Phase 8 Claimed: {best_result['accuracy'] - phase8_claimed:+.4f} ({((best_result['accuracy'] - phase8_claimed)/phase8_claimed)*100:+.2f}%)")
    
    # Data leakage analysis
    phase8_improvement = phase8_claimed - phase6_baseline
    true_improvement = best_result['accuracy'] - phase6_baseline
    leakage_impact = phase8_improvement - true_improvement
    
    print(f"\n🚨 DATA LEAKAGE ANALYSIS:")
    print("=" * 60)
    print(f"Phase 8 claimed improvement: +{phase8_improvement:.4f} ({phase8_improvement*100:.2f} percentage points)")
    print(f"True improvement: +{true_improvement:.4f} ({true_improvement*100:.2f} percentage points)")
    print(f"Data leakage impact: +{leakage_impact:.4f} ({leakage_impact*100:.2f} percentage points)")
    print(f"Leakage percentage: {(leakage_impact/phase8_improvement)*100:.1f}% of claimed improvement was due to leakage")
    
    # Generate test predictions with best model
    print(f"\n🧪 Generating test predictions with best model...")
    X_test = preprocessor.transform(test_data)
    
    if 'Ensemble' in best_result['model_name']:
        test_predictions = simple_ensemble_predict(loaded_models, X_test)
    else:
        # Find the best individual model
        best_model = None
        for i, result in enumerate(all_results):
            if result['model_name'] == best_result['model_name']:
                best_model = loaded_models[i][0]
                break
        
        if best_model:
            best_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                test_predictions = best_model(X_test_tensor).squeeze().numpy()
    
    # Save corrected results
    phase9_results = {
        'timestamp': start_time.isoformat(),
        'validation_method': 'stratified_85_15_split',
        'validation_set_size': len(X_val),
        'train_set_size': len(X_train),
        'models_evaluated': len(loaded_models),
        'validation_results': all_results,
        'best_model': {
            'name': best_result['model_name'],
            'architecture': best_result.get('architecture', 'Unknown'),
            'accuracy': best_result['accuracy'],
            'f1_score': best_result['f1_score'],
            'precision': best_result['precision'],
            'recall': best_result['recall']
        },
        'baseline_comparisons': {
            'phase6_baseline': phase6_baseline,
            'phase8_claimed': phase8_claimed,
            'true_improvement_vs_phase6': true_improvement,
            'phase8_claimed_improvement': phase8_improvement,
            'data_leakage_impact': leakage_impact,
            'leakage_percentage': (leakage_impact/phase8_improvement)*100
        },
        'test_predictions': {
            'count': len(test_predictions),
            'mean': float(np.mean(test_predictions)),
            'std': float(np.std(test_predictions)),
            'min': float(np.min(test_predictions)),
            'max': float(np.max(test_predictions)),
            'positive_predictions': int(np.sum(test_predictions > 0.5)),
            'negative_predictions': int(np.sum(test_predictions <= 0.5))
        }
    }
    
    # Save corrected results
    with open('experiments/phase9_corrected_validation.json', 'w') as f:
        json.dump(phase9_results, f, indent=2)
    
    # Save corrected test predictions
    test_df = pd.DataFrame({
        'prediction': test_predictions,
        'binary_prediction': (test_predictions > 0.5).astype(int)
    })
    test_df.to_csv('experiments/phase9_corrected_test_predictions.csv', index=False)
    
    # Print summary table
    print(f"\n📊 Phase 9 Corrected Model Comparison:")
    print("-" * 90)
    print(f"{'Model':<30} {'Architecture':<30} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 90)
    for result in all_results:
        arch = result.get('architecture', 'Unknown')[:28] + '..' if len(result.get('architecture', 'Unknown')) > 30 else result.get('architecture', 'Unknown')
        print(f"{result['model_name']:<30} {arch:<30} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} {result['precision']:<10.4f} {result['recall']:<10.4f}")
    print("-" * 90)
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Phase 9 corrected validation completed in {elapsed_time:.1f}s")
    print(f"💾 Corrected results saved to: experiments/phase9_corrected_validation.json")
    print(f"💾 Corrected test predictions: experiments/phase9_corrected_test_predictions.csv")
    
    # Achievement assessment
    target_accuracy = 0.95
    gap_to_target = target_accuracy - best_result['accuracy']
    
    print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS:")
    print("=" * 60)
    print(f"Current best accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"Target accuracy: {target_accuracy:.4f} ({target_accuracy*100:.2f}%)")
    print(f"Gap to target: {gap_to_target:.4f} ({gap_to_target*100:.2f} percentage points)")
    print(f"Relative gap: {(gap_to_target/target_accuracy)*100:.1f}% improvement needed")
    
    if gap_to_target > 0.25:
        print("❌ Target appears unrealistic with current approach")
    elif gap_to_target > 0.10:
        print("⚠️  Target is challenging but potentially achievable with advanced methods")
    else:
        print("✅ Target is achievable with optimization")
    
    return phase9_results

if __name__ == "__main__":
    results = main() 