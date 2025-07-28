#!/usr/bin/env python3
"""
Train Tree-Based Models - Phase 7
XGBoost, LightGBM, and Random Forest with hyperparameter optimization
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from data_loader import load_pickle_data
from tree_models import TreeModelManager, XGBoostModel, LightGBMModel, RandomForestModel
import pickle

def prepare_data():
    """
    Load and prepare data for tree-based models
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val, X_test, preprocessor)
    """
    print("📊 Loading and preparing data for tree models...")
    
    # Load data
    train_data = load_pickle_data('data/train.pickle')
    test_data = load_pickle_data('data/test.pickle')
    
    # Load preprocessor
    with open('models/final_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Prepare training data
    X_train_full = preprocessor.transform(train_data.drop('is_correct', axis=1))
    y_train_full = train_data['is_correct'].values.astype(int)
    
    # Split for training/validation (80/20 split for tree models)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )
    
    # Prepare test data
    X_test = preprocessor.transform(test_data)
    
    print(f"✅ Data prepared:")
    print(f"  📈 Training: {X_train.shape[0]:,} samples")
    print(f"  📊 Validation: {X_val.shape[0]:,} samples") 
    print(f"  🧪 Test: {X_test.shape[0]:,} samples")
    print(f"  🔢 Features: {X_train.shape[1]} features")
    
    return X_train, X_val, y_train, y_val, X_test, preprocessor

def train_individual_models(X_train, X_val, y_train, y_val, optimize_hyperparams=True):
    """
    Train individual tree models separately for detailed analysis
    
    Args:
        X_train, X_val, y_train, y_val: Training and validation data
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with individual model results
    """
    print("🌲 Training individual tree-based models...")
    
    individual_results = {}
    
    # XGBoost
    print("\n1️⃣ Training XGBoost...")
    xgb_model = XGBoostModel(random_state=42)
    
    if optimize_hyperparams:
        xgb_results = xgb_model.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=40
        )
    else:
        xgb_model.fit(X_train, y_train, X_val, y_val)
        val_pred = xgb_model.predict(X_val)
        val_pred_proba = xgb_model.predict_proba(X_val)
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        xgb_results = {
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_pred_proba)
        }
    
    individual_results['xgboost'] = {
        'model': xgb_model,
        'results': xgb_results
    }
    
    # LightGBM
    print("\n2️⃣ Training LightGBM...")
    lgb_model = LightGBMModel(random_state=42)
    
    if optimize_hyperparams:
        lgb_results = lgb_model.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=40
        )
    else:
        lgb_model.fit(X_train, y_train, X_val, y_val)
        val_pred = lgb_model.predict(X_val)
        val_pred_proba = lgb_model.predict_proba(X_val)
        
        lgb_results = {
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_pred_proba)
        }
    
    individual_results['lightgbm'] = {
        'model': lgb_model,
        'results': lgb_results
    }
    
    # Random Forest
    print("\n3️⃣ Training Random Forest...")
    rf_model = RandomForestModel(random_state=42)
    
    if optimize_hyperparams:
        rf_results = rf_model.optimize_hyperparameters(
            X_train, y_train, X_val, y_val, n_trials=25
        )
    else:
        rf_model.fit(X_train, y_train)
        val_pred = rf_model.predict(X_val)
        val_pred_proba = rf_model.predict_proba(X_val)
        
        rf_results = {
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_roc_auc': roc_auc_score(y_val, val_pred_proba)
        }
    
    individual_results['random_forest'] = {
        'model': rf_model,
        'results': rf_results
    }
    
    return individual_results

def analyze_feature_importance(individual_results, feature_names):
    """
    Analyze and compare feature importance across models
    
    Args:
        individual_results: Dictionary with trained models
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importance comparison
    """
    print("🔍 Analyzing feature importance across models...")
    
    importance_data = []
    
    for model_name, model_info in individual_results.items():
        model = model_info['model']
        
        try:
            importance_df = model.get_feature_importance(feature_names)
            for _, row in importance_df.iterrows():
                importance_data.append({
                    'model': model_name,
                    'feature': row['feature'],
                    'importance': row['importance']
                })
        except Exception as e:
            print(f"⚠️  Could not get feature importance for {model_name}: {e}")
    
    if importance_data:
        importance_comparison = pd.DataFrame(importance_data)
        
        # Create pivot table for comparison
        pivot_importance = importance_comparison.pivot(
            index='feature', columns='model', values='importance'
        ).fillna(0)
        
        # Calculate average importance across models
        pivot_importance['average'] = pivot_importance.mean(axis=1)
        pivot_importance = pivot_importance.sort_values('average', ascending=False)
        
        print("📊 Top 10 Most Important Features (Average):")
        print(pivot_importance.head(10).round(4))
        
        return pivot_importance
    
    return None

def generate_test_predictions(individual_results, X_test):
    """
    Generate test predictions with all models
    
    Args:
        individual_results: Dictionary with trained models
        X_test: Test features
        
    Returns:
        Dictionary with test predictions for each model
    """
    print("🧪 Generating test predictions with all tree models...")
    
    test_predictions = {}
    
    for model_name, model_info in individual_results.items():
        model = model_info['model']
        
        try:
            pred_proba = model.predict_proba(X_test)
            pred_binary = model.predict(X_test)
            
            test_predictions[model_name] = {
                'probabilities': pred_proba.tolist(),
                'binary_predictions': pred_binary.tolist(),
                'stats': {
                    'mean_probability': float(np.mean(pred_proba)),
                    'std_probability': float(np.std(pred_proba)),
                    'positive_predictions': int(np.sum(pred_binary)),
                    'negative_predictions': int(len(pred_binary) - np.sum(pred_binary))
                }
            }
            
            print(f"✅ {model_name}: {test_predictions[model_name]['stats']}")
            
        except Exception as e:
            print(f"❌ Error generating predictions for {model_name}: {e}")
    
    return test_predictions

def train_tree_models():
    """
    Main function to train all tree-based models
    """
    print("🚀 Phase 7: Training Tree-Based Models")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Prepare data
        X_train, X_val, y_train, y_val, X_test, preprocessor = prepare_data()
        
        # Get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names'):
            feature_names = preprocessor.get_feature_names()
        else:
            # Create generic feature names
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Train models using TreeModelManager
        print("\n🔧 Training all models with TreeModelManager...")
        manager = TreeModelManager(random_state=42)
        manager_results = manager.train_all_models(
            X_train, y_train, X_val, y_val,
            optimize_hyperparams=True,
            n_trials=40
        )
        
        # Train individual models for detailed analysis
        print("\n🌲 Training individual models for detailed analysis...")
        individual_results = train_individual_models(
            X_train, X_val, y_train, y_val, optimize_hyperparams=True
        )
        
        # Compare with neural network baseline
        baseline_accuracy = 0.6424  # Phase 6 best accuracy
        best_tree_result = max(
            [(name, res['results']) for name, res in individual_results.items()],
            key=lambda x: x[1]['val_accuracy']
        )
        best_tree_name, best_tree_metrics = best_tree_result
        best_tree_accuracy = best_tree_metrics['val_accuracy']
        
        improvement = best_tree_accuracy - baseline_accuracy
        improvement_pct = (improvement / baseline_accuracy) * 100
        
        print(f"\n🏆 Best Tree Model: {best_tree_name.upper()}")
        print(f"📊 Accuracy: {best_tree_accuracy:.4f}")
        print(f"📈 Improvement over NN baseline: +{improvement:.4f} ({improvement_pct:+.2f}%)")
        
        # Analyze feature importance
        feature_importance_df = analyze_feature_importance(individual_results, feature_names)
        
        # Generate test predictions
        test_predictions = generate_test_predictions(individual_results, X_test)
        
        # Save models
        print("\n💾 Saving trained models...")
        manager.save_models('models')
        
        # Save individual models
        for model_name, model_info in individual_results.items():
            model_path = f'models/tree_{model_name}_individual.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model_info['model'], f)
            print(f"✅ Saved {model_name} to {model_path}")
        
        # Prepare comprehensive results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_data_size': len(X_train),
            'validation_data_size': len(X_val),
            'test_data_size': len(X_test),
            'num_features': X_train.shape[1],
            'baseline_nn_accuracy': baseline_accuracy,
            'best_tree_model': best_tree_name,
            'best_tree_accuracy': best_tree_accuracy,
            'improvement_over_baseline': improvement,
            'improvement_percentage': improvement_pct,
            'individual_model_results': {
                name: info['results'] for name, info in individual_results.items()
            },
            'manager_results': manager_results,
            'test_predictions': test_predictions,
            'training_time_minutes': (time.time() - start_time) / 60
        }
        
        # Add feature importance if available
        if feature_importance_df is not None:
            results_summary['feature_importance'] = {
                'top_features': feature_importance_df.head(10).to_dict(),
                'all_features': feature_importance_df.to_dict()
            }
        
        # Save comprehensive results
        results_file = 'experiments/phase7_tree_models_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Save test predictions CSV
        if test_predictions:
            test_df_data = {}
            for model_name, preds in test_predictions.items():
                test_df_data[f'{model_name}_probability'] = preds['probabilities']
                test_df_data[f'{model_name}_prediction'] = preds['binary_predictions']
            
            test_df = pd.DataFrame(test_df_data)
            test_df.to_csv('experiments/phase7_tree_test_predictions.csv', index=False)
            print("📄 Test predictions saved to CSV")
        
        # Print comparison table
        print("\n📊 Tree Models Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<15} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<11} {'Recall':<10} {'ROC AUC':<10}")
        print("-" * 80)
        
        # Add baseline for comparison
        print(f"{'NN Baseline':<15} {baseline_accuracy:<10.4f} {'--':<10} {'--':<11} {'--':<10} {'--':<10}")
        
        for model_name, model_info in individual_results.items():
            results = model_info['results']
            print(f"{model_name.title():<15} {results['val_accuracy']:<10.4f} {results['val_f1']:<10.4f} {results['val_precision']:<11.4f} {results['val_recall']:<10.4f} {results['val_roc_auc']:<10.4f}")
        print("-" * 80)
        
        training_time = time.time() - start_time
        print(f"\n🏁 Tree models training completed in {training_time:.2f}s")
        print(f"🎉 Phase 7 Tree Models: SUCCESS")
        
        return results_summary
        
    except Exception as e:
        print(f"❌ Error in tree models training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run tree models training
    results = train_tree_models()
    
    if results:
        print(f"\n✅ Tree models training completed successfully!")
        print(f"🏆 Best tree model: {results['best_tree_model']}")
        print(f"📊 Best accuracy: {results['best_tree_accuracy']:.4f}")
        print(f"📈 Improvement: +{results['improvement_percentage']:.2f}%")
    else:
        print("❌ Tree models training failed!")
        sys.exit(1) 