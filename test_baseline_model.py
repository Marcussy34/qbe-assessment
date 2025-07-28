"""
Test script for baseline model training and evaluation.
Validates that all Phase 4 components work together correctly.
"""

import sys
sys.path.append('src')

import torch
from utils import set_random_seeds, get_device, print_system_info
from data_loader import load_pickle_data
from preprocessor import DataPreprocessor, create_data_splits, create_data_loaders
from model import create_model, model_summary
from train import ModelTrainer, get_default_configs
from evaluate import ModelEvaluator

def test_baseline_model():
    """Test the complete baseline model pipeline."""
    print("=== TESTING BASELINE MODEL PIPELINE ===\n")
    
    # Set up reproducibility
    set_random_seeds(42)
    print_system_info()
    device = get_device()
    
    # Load and preprocess data
    print("\n=== LOADING AND PREPROCESSING DATA ===")
    train_df = load_pickle_data("data/train.pickle")
    
    if train_df is None:
        print("Failed to load training data")
        return False
    
    # Use small subset for quick testing
    print("Using subset of data for quick testing...")
    train_subset = train_df.sample(n=10000, random_state=42)
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_processed, y = preprocessor.fit_transform(train_subset)
    
    # Create train/val splits
    X_train, X_val, y_train, y_val = create_data_splits(X_processed, y, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, batch_size=64)
    
    print(f"Data prepared: {X_train.shape[0]} train, {X_val.shape[0]} val samples")
    
    # Test model creation
    print("\n=== TESTING MODEL CREATION ===")
    
    # Test different model types
    model_types = ['baseline', 'optimized', 'deep', 'wide']
    
    for model_type in model_types:
        try:
            model = create_model(model_type, input_dim=14)
            print(f"✅ {model_type} model created successfully")
            
            # Test model summary
            if model_type == 'baseline':  # Only show detailed summary for baseline
                model_summary(model)
            else:
                print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
                
        except Exception as e:
            print(f"❌ {model_type} model creation failed: {e}")
            return False
    
    # Test baseline model training
    print("\n=== TESTING BASELINE MODEL TRAINING ===")
    
    baseline_model = create_model('baseline', input_dim=14, hidden_dim=64, dropout_rate=0.2)
    
    # Test trainer initialization
    try:
        trainer = ModelTrainer(
            model=baseline_model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.001,
            optimizer_type='adam',
            scheduler_type='plateau',
            early_stopping_patience=5
        )
        print("✅ Trainer initialized successfully")
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        return False
    
    # Test short training run
    try:
        print("Running short training (5 epochs)...")
        history = trainer.train(num_epochs=5, print_every=2)
        print("✅ Training completed successfully")
        
        # Check training history
        print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Test evaluation
    print("\n=== TESTING MODEL EVALUATION ===")
    
    try:
        evaluator = ModelEvaluator(trainer.model, device)
        
        # Test prediction
        predictions, targets = evaluator.predict(val_loader)
        print(f"✅ Predictions generated: {len(predictions)} samples")
        
        # Test metrics calculation
        metrics = evaluator.calculate_metrics(val_loader)
        print(f"✅ Metrics calculated:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    # Test configuration system
    print("\n=== TESTING CONFIGURATION SYSTEM ===")
    
    try:
        configs = get_default_configs()
        print(f"✅ Default configurations loaded: {list(configs.keys())}")
        
        # Test config usage
        baseline_config = configs['baseline']
        print(f"✅ Baseline config: {baseline_config}")
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False
    
    # Test model saving/loading
    print("\n=== TESTING MODEL PERSISTENCE ===")
    
    try:
        # Save model
        model_path = "models/test_baseline_model.pth"
        torch.save(trainer.model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Load model
        loaded_model = create_model('baseline', input_dim=14, hidden_dim=64, dropout_rate=0.2)
        loaded_model.load_state_dict(torch.load(model_path))
        print("✅ Model loaded successfully")
        
        # Test that loaded model gives same predictions
        loaded_model.eval()
        with torch.no_grad():
            test_input = torch.randn(1, 14)
            orig_output = trainer.model(test_input)
            loaded_output = loaded_model(test_input)
            
            if torch.allclose(orig_output, loaded_output, atol=1e-6):
                print("✅ Model persistence verified")
            else:
                print("❌ Model persistence failed - outputs don't match")
                return False
                
    except Exception as e:
        print(f"❌ Model persistence failed: {e}")
        return False
    
    # Summary
    print(f"\n=== BASELINE MODEL TEST SUMMARY ===")
    print(f"✅ Data preprocessing: PASSED")
    print(f"✅ Model creation (all types): PASSED")
    print(f"✅ Training pipeline: PASSED")
    print(f"✅ Evaluation utilities: PASSED")
    print(f"✅ Configuration system: PASSED")
    print(f"✅ Model persistence: PASSED")
    
    # Save test results
    with open("experiments/baseline_test_results.txt", "w") as f:
        f.write("=== BASELINE MODEL TEST RESULTS ===\n\n")
        f.write(f"Training subset: {len(train_subset)} samples\n")
        f.write(f"Final validation accuracy: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters() if p.requires_grad):,}\n")
        f.write(f"Test evaluation metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"All tests: PASSED\n")
    
    print(f"\nTest results saved to experiments/baseline_test_results.txt")
    
    return True

if __name__ == "__main__":
    success = test_baseline_model()
    if success:
        print("\n🎉 ALL BASELINE MODEL TESTS PASSED!")
        print("Ready to proceed to full training and experimentation!")
    else:
        print("\n❌ BASELINE MODEL TESTS FAILED!")
        sys.exit(1) 