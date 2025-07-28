#!/usr/bin/env python3
"""
Train Ensemble Models - Phase 7
Combines existing neural network models to create ensemble predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import time
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append('src')

from data_loader import load_pickle_data
from preprocessor import DataPreprocessor
from ensemble import EnsembleManager, SimpleEnsemble, StackingEnsemble
import pickle

def load_existing_models():
    """
    Load existing trained neural network models from previous phases
    
    Returns:
        List of model file paths that exist
    """
    model_paths = [
        'models/final_optimized_model.pth',
        'models/test_baseline_model.pth'
    ]
    
    # Check which models exist
    existing_paths = []
    for path in model_paths:
        if Path(path).exists():
            existing_paths.append(path)
            print(f"✅ Found model: {path}")
        else:
            print(f"⚠️  Model not found: {path}")
    
    if not existing_paths:
        raise FileNotFoundError("No existing models found. Please train base models first.")
    
    return existing_paths

def create_diverse_models():
    """
    Create additional neural network models with different configurations
    for more diverse ensemble
    
    Returns:
        List of newly trained model paths
    """
    print("🧠 Creating diverse neural network models for ensemble...")
    
    # Import required modules
    from model import BaselineNet, OptimizedNet
    from torch.utils.data import TensorDataset, DataLoader
    import torch.optim as optim
    import torch.nn as nn
    
    # Load data
    print("📊 Loading training data...")
    from data_loader import load_pickle_data
    train_data = load_pickle_data('data/train.pickle')
    test_data = load_pickle_data('data/test.pickle')
    
    # Load preprocessor
    with open('models/final_preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Prepare data
    X_train = preprocessor.transform(train_data.drop('is_correct', axis=1))
    y_train = train_data['is_correct'].values.astype(float)
    
    # Split for training/validation
    from sklearn.model_selection import train_test_split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_split)
    y_train_tensor = torch.FloatTensor(y_train_split)
    X_val_tensor = torch.FloatTensor(X_val_split)
    y_val_tensor = torch.FloatTensor(y_val_split)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device('cpu')  # Use CPU for consistency
    diverse_models = []
    
    # Model 1: Different baseline architecture
    print("🔨 Training diverse baseline model...")
    model1 = BaselineNet(input_dim=14, hidden_dim=128)
    
    # Simple training loop for model1
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    model1.train()
    for epoch in range(20):  # Fewer epochs for speed
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer1.zero_grad()
            outputs = model1(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer1.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    model1_path = 'models/ensemble_baseline_128.pth'
    torch.save(model1.state_dict(), model1_path)
    diverse_models.append(model1_path)
    print(f"💾 Saved model: {model1_path}")
    
    # Model 2: Optimized with different dropout
    print("🔨 Training diverse optimized model...")
    model2 = OptimizedNet(input_dim=14, hidden_dims=[256, 128, 64], dropout_rate=0.3)
    
    # Simple training loop for model2
    optimizer2 = optim.Adam(model2.parameters(), lr=0.0005)
    
    model2.train()
    for epoch in range(20):  # Fewer epochs for speed
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer2.zero_grad()
            outputs = model2(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer2.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    model2_path = 'models/ensemble_optimized_high_dropout.pth'
    torch.save(model2.state_dict(), model2_path)
    diverse_models.append(model2_path)
    print(f"💾 Saved model: {model2_path}")
    
    # Model 3: Wide network with different learning rate
    print("🔨 Training diverse wide model...")
    model3 = BaselineNet(input_dim=14, hidden_dim=512)  # Wider network
    
    # Simple training loop for model3
    optimizer3 = optim.Adam(model3.parameters(), lr=0.0001)  # Lower LR
    
    model3.train()
    for epoch in range(20):  # Fewer epochs for speed
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer3.zero_grad()
            outputs = model3(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer3.step()
            total_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")
    
    model3_path = 'models/ensemble_wide_512.pth'
    torch.save(model3.state_dict(), model3_path)
    diverse_models.append(model3_path)
    print(f"💾 Saved model: {model3_path}")
    
    return diverse_models

def train_ensemble_models():
    """
    Main function to train ensemble models
    """
    print("🚀 Phase 7: Training Ensemble Models")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Load existing models
        existing_models = load_existing_models()
        
        # Create additional diverse models if we have less than 3
        if len(existing_models) < 3:
            print(f"📈 Only {len(existing_models)} existing models found. Creating diverse models...")
            diverse_models = create_diverse_models()
            all_model_paths = existing_models + diverse_models
        else:
            all_model_paths = existing_models
        
        print(f"🎯 Using {len(all_model_paths)} models for ensemble:")
        for i, path in enumerate(all_model_paths, 1):
            print(f"  {i}. {path}")
        
        # Load data for ensemble training
        print("\n📊 Loading data for ensemble training...")
        from data_loader import load_pickle_data
        train_data = load_pickle_data('data/train.pickle')
        test_data = load_pickle_data('data/test.pickle')
        
        # Load preprocessor
        with open('models/final_preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        # Prepare data
        X_train = preprocessor.transform(train_data.drop('is_correct', axis=1))
        y_train = train_data['is_correct'].values
        
        # Split for ensemble training/validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_split)
        X_val_tensor = torch.FloatTensor(X_val_split)
        
        # Initialize ensemble manager
        print("\n🔧 Initializing ensemble manager...")
        ensemble_manager = EnsembleManager(all_model_paths)
        
        # Compare different ensemble methods
        print("\n📊 Comparing ensemble methods...")
        ensemble_results = ensemble_manager.compare_ensembles(X_val_tensor, y_val_split)
        
        # Train and evaluate stacking ensemble
        print("\n🏗️ Training stacking ensemble...")
        stacking_ensemble = ensemble_manager.create_stacking_ensemble()
        stacking_ensemble.fit(X_train_tensor, y_train_split)
        
        # Evaluate stacking ensemble
        stacking_results = ensemble_manager.evaluate_ensemble(
            stacking_ensemble, X_val_tensor, y_val_split, 'stacking_ensemble'
        )
        
        # Find best ensemble method
        all_results = {**ensemble_results, 'stacking_ensemble': stacking_results}
        best_method = max(all_results.keys(), key=lambda x: all_results[x]['accuracy'])
        best_accuracy = all_results[best_method]['accuracy']
        
        print(f"\n🏆 Best ensemble method: {best_method}")
        print(f"🎯 Best accuracy: {best_accuracy:.4f}")
        
        # Save best ensemble
        if best_method == 'stacking_ensemble':
            best_ensemble = stacking_ensemble
        else:
            best_ensemble = ensemble_manager.create_simple_ensemble()
        
        ensemble_manager.save_best_ensemble(
            best_ensemble, best_method, 'models/best_ensemble.pkl'
        )
        
        # Calculate improvement over single model baseline
        baseline_accuracy = 0.6424  # Phase 6 best accuracy
        improvement = best_accuracy - baseline_accuracy
        improvement_pct = (improvement / baseline_accuracy) * 100
        
        print(f"\n📈 Improvement Analysis:")
        print(f"  🔹 Phase 6 baseline: {baseline_accuracy:.4f}")
        print(f"  🔹 Best ensemble: {best_accuracy:.4f}")
        print(f"  🔹 Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
        
        # Prepare final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'model_paths': all_model_paths,
            'ensemble_methods': all_results,
            'best_method': best_method,
            'best_accuracy': best_accuracy,
            'baseline_accuracy': baseline_accuracy,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
            'training_time_minutes': (time.time() - start_time) / 60
        }
        
        # Save results
        results_file = 'experiments/phase7_ensemble_results.json'
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Print summary table
        print("\n📊 Ensemble Methods Comparison:")
        print("-" * 60)
        print(f"{'Method':<20} {'Accuracy':<10} {'F1 Score':<10} {'ROC AUC':<10}")
        print("-" * 60)
        for method, results in all_results.items():
            print(f"{method:<20} {results['accuracy']:<10.4f} {results['f1_score']:<10.4f} {results['roc_auc']:<10.4f}")
        print("-" * 60)
        
        # Test on full test set if possible
        print("\n🧪 Generating test predictions with best ensemble...")
        try:
            X_test = preprocessor.transform(test_data)
            X_test_tensor = torch.FloatTensor(X_test)
            
            test_predictions = best_ensemble.predict_proba(X_test_tensor)
            
            # Save test predictions
            test_results = {
                'test_predictions': test_predictions.tolist(),
                'ensemble_method': best_method,
                'prediction_stats': {
                    'mean': float(np.mean(test_predictions)),
                    'std': float(np.std(test_predictions)),
                    'min': float(np.min(test_predictions)),
                    'max': float(np.max(test_predictions)),
                    'positive_predictions': int(np.sum(test_predictions > 0.5)),
                    'negative_predictions': int(np.sum(test_predictions <= 0.5))
                }
            }
            
            with open('experiments/phase7_test_predictions.json', 'w') as f:
                json.dump(test_results, f, indent=2)
            
            # Save CSV format
            test_df = pd.DataFrame({
                'prediction': test_predictions,
                'binary_prediction': (test_predictions > 0.5).astype(int)
            })
            test_df.to_csv('experiments/phase7_test_predictions.csv', index=False)
            
            print(f"✅ Test predictions saved")
            print(f"📊 Test stats: {test_results['prediction_stats']}")
            
        except Exception as e:
            print(f"⚠️  Error generating test predictions: {e}")
        
        training_time = time.time() - start_time
        print(f"\n🏁 Ensemble training completed in {training_time:.2f}s")
        print(f"🎉 Phase 7 Ensemble Models: SUCCESS")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Error in ensemble training: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run ensemble training
    results = train_ensemble_models()
    
    if results:
        print(f"\n✅ Ensemble training completed successfully!")
        print(f"🏆 Best accuracy: {results['best_accuracy']:.4f}")
        print(f"📈 Improvement: +{results['improvement_percentage']:.2f}%")
    else:
        print("❌ Ensemble training failed!")
        sys.exit(1) 