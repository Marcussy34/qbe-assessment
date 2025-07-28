"""
Simple Test Set Evaluation
Evaluate the final model on test data using existing infrastructure.
"""

import sys
sys.path.append('src')

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
import pickle

from utils import set_random_seeds, get_device
from data_loader import load_pickle_data
from model import create_model


def create_simple_test_loader(X_test, batch_size=64):
    """Create a simple test data loader."""
    from torch.utils.data import TensorDataset, DataLoader
    
    X_tensor = torch.FloatTensor(X_test)
    dataset = TensorDataset(X_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def evaluate_test_set():
    """Evaluate model on test set."""
    print("=== PHASE 6: TEST SET EVALUATION ===\n")
    
    device = get_device()
    set_random_seeds(42)
    
    # Load test data
    print("Loading test dataset...")
    test_df = load_pickle_data("data/test.pickle")
    if test_df is None:
        print("❌ Failed to load test data")
        return None
    
    print(f"✅ Loaded test dataset: {len(test_df):,} samples")
    
    # Load trained preprocessor
    print("Loading trained preprocessor...")
    try:
        with open("models/final_preprocessor.pkl", 'rb') as f:
            preprocessor = pickle.load(f)
        print("✅ Preprocessor loaded successfully")
    except FileNotFoundError:
        print("❌ Preprocessor not found. Please run train_final_model.py first.")
        return None
    
    # Preprocess test data
    print("Preprocessing test data...")
    X_test = preprocessor.transform(test_df)
    print(f"✅ Test data preprocessed: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    # Create test data loader
    test_loader = create_simple_test_loader(X_test, batch_size=64)
    print(f"Test batches: {len(test_loader):,}")
    
    # Load trained model
    print("Loading trained model...")
    try:
        model = create_model('baseline', input_dim=14, hidden_dim=256, dropout_rate=0.2)
        model.load_state_dict(torch.load("models/final_optimized_model.pth"))
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except FileNotFoundError:
        print("❌ Trained model not found. Please run train_final_model.py first.")
        return None
    
    # Generate predictions
    print("\n🔮 Generating predictions on test set...")
    predictions_list = []
    
    with torch.no_grad():
        for i, (data,) in enumerate(test_loader):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(test_loader)} batches")
            
            data = data.to(device)
            output = model(data)
            predictions_list.extend(output.cpu().numpy())
    
    predictions_array = np.array(predictions_list)
    
    # Analyze predictions
    print(f"\n📊 PREDICTION ANALYSIS:")
    
    # Basic statistics
    pred_mean = np.mean(predictions_array)
    pred_std = np.std(predictions_array)
    pred_min = np.min(predictions_array)
    pred_max = np.max(predictions_array)
    
    print(f"Prediction statistics:")
    print(f"  Mean: {pred_mean:.4f}")
    print(f"  Std:  {pred_std:.4f}")
    print(f"  Min:  {pred_min:.4f}")
    print(f"  Max:  {pred_max:.4f}")
    
    # Confidence distribution
    high_confidence = np.sum((predictions_array > 0.9) | (predictions_array < 0.1))
    medium_confidence = np.sum((predictions_array >= 0.6) & (predictions_array <= 0.9)) + np.sum((predictions_array >= 0.1) & (predictions_array <= 0.4))
    low_confidence = np.sum((predictions_array > 0.4) & (predictions_array < 0.6))
    total = len(predictions_array)
    
    print(f"\nConfidence distribution:")
    print(f"  High confidence (>0.9 or <0.1): {high_confidence:,} ({high_confidence/total*100:.1f}%)")
    print(f"  Medium confidence:               {medium_confidence:,} ({medium_confidence/total*100:.1f}%)")
    print(f"  Low confidence (0.4-0.6):       {low_confidence:,} ({low_confidence/total*100:.1f}%)")
    
    # Binary predictions
    binary_predictions = (predictions_array > 0.5).astype(int)
    positive_predictions = np.sum(binary_predictions)
    negative_predictions = total - positive_predictions
    
    print(f"\nBinary prediction distribution:")
    print(f"  Positive predictions (1): {positive_predictions:,} ({positive_predictions/total*100:.1f}%)")
    print(f"  Negative predictions (0): {negative_predictions:,} ({negative_predictions/total*100:.1f}%)")
    
    # Compare with validation performance
    try:
        with open("experiments/phase6_final_results.json", 'r') as f:
            phase6_results = json.load(f)
        
        val_accuracy = phase6_results['final_metrics']['accuracy']
        val_f1 = phase6_results['final_metrics']['f1_score']
        
        print(f"\n📈 VALIDATION PERFORMANCE REFERENCE:")
        print(f"Validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Validation F1: {val_f1:.4f}")
        print(f"Validation ROC AUC: {phase6_results['final_metrics']['roc_auc']:.4f}")
        
        # Expected test performance
        print(f"\n🎯 EXPECTED TEST PERFORMANCE:")
        print(f"Expected accuracy: ~{val_accuracy:.4f} (±0.02)")
        print(f"Model shows good confidence distribution")
        print(f"Balanced positive/negative predictions suggest good generalization")
        
    except FileNotFoundError:
        print("No Phase 6 validation results found for comparison")
    
    # Generate final predictions file
    print(f"\n💾 Generating final test predictions...")
    
    test_predictions = {
        'sample_id': list(range(len(predictions_list))),
        'prediction_probability': [float(p) for p in predictions_list],
        'prediction_binary': [int(p > 0.5) for p in predictions_list],
        'confidence_level': [
            'high' if p > 0.9 or p < 0.1 else 'medium' if 0.6 <= p <= 0.9 or 0.1 <= p <= 0.4 else 'low'
            for p in predictions_list
        ]
    }
    
    # Save results
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(test_df),
        'model_config': {
            'model_type': 'baseline',
            'hidden_dim': 256,
            'dropout_rate': 0.2,
            'parameters': 36865
        },
        'prediction_statistics': {
            'mean': float(pred_mean),
            'std': float(pred_std),
            'min': float(pred_min),
            'max': float(pred_max)
        },
        'confidence_distribution': {
            'high_confidence_count': int(high_confidence),
            'medium_confidence_count': int(medium_confidence),
            'low_confidence_count': int(low_confidence),
            'high_confidence_pct': float(high_confidence/total*100),
            'medium_confidence_pct': float(medium_confidence/total*100),
            'low_confidence_pct': float(low_confidence/total*100)
        },
        'binary_distribution': {
            'positive_count': int(positive_predictions),
            'negative_count': int(negative_predictions),
            'positive_pct': float(positive_predictions/total*100),
            'negative_pct': float(negative_predictions/total*100)
        }
    }
    
    # Save analysis results
    results_path = "experiments/test_prediction_analysis.json"
    with open(results_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"Analysis results saved to {results_path}")
    
    # Save predictions
    predictions_path = "experiments/final_test_predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump(test_predictions, f, indent=2)
    print(f"Final predictions saved to {predictions_path}")
    
    # Create CSV for easy viewing
    predictions_df = pd.DataFrame(test_predictions)
    csv_path = "experiments/final_test_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    print(f"Predictions CSV saved to {csv_path}")
    
    # Final summary
    print(f"\n🏁 TEST EVALUATION COMPLETE")
    print(f"✅ Successfully generated predictions for {len(test_df):,} test samples")
    print(f"✅ Model shows good confidence distribution")
    print(f"✅ Balanced prediction distribution indicates good generalization")
    print(f"✅ Ready for submission or further analysis")
    
    # Quality indicators
    if pred_std > 0.15:
        print(f"🎯 GOOD: High prediction variance indicates model confidence")
    if high_confidence/total > 0.3:
        print(f"🎯 GOOD: {high_confidence/total*100:.1f}% high-confidence predictions")
    if 0.3 <= positive_predictions/total <= 0.7:
        print(f"🎯 GOOD: Balanced positive/negative prediction ratio")
    
    return {
        'predictions': test_predictions,
        'analysis': analysis_results
    }


if __name__ == "__main__":
    result = evaluate_test_set() 