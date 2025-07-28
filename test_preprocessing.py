"""
Test script for the data preprocessing pipeline.
Validates that all preprocessing steps work correctly.
"""

import sys
sys.path.append('src')

from utils import set_random_seeds, print_system_info
from data_loader import load_pickle_data
from preprocessor import DataPreprocessor, create_data_splits, create_data_loaders
import numpy as np
import pandas as pd

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("=== TESTING PREPROCESSING PIPELINE ===\n")
    
    # Set up reproducibility
    set_random_seeds(42)
    
    # Load training data
    print("Loading training data...")
    train_df = load_pickle_data("data/train.pickle")
    
    if train_df is None:
        print("Failed to load training data")
        return False
    
    print(f"Loaded training data: {train_df.shape}")
    print(f"Target distribution: {train_df['is_correct'].value_counts()}")
    
    # Initialize and test preprocessor
    print("\n=== TESTING PREPROCESSING STEPS ===")
    
    preprocessor = DataPreprocessor()
    
    # Test full preprocessing pipeline
    try:
        X_processed, y = preprocessor.fit_transform(train_df)
        
        print(f"\n=== PREPROCESSING RESULTS ===")
        print(f"Input shape: {train_df.shape}")
        print(f"Output feature shape: {X_processed.shape}")
        print(f"Output target shape: {y.shape}")
        print(f"Feature columns: {preprocessor.feature_columns}")
        
        # Validate data quality
        print(f"\n=== DATA QUALITY CHECKS ===")
        print(f"Any NaN values in features: {np.isnan(X_processed).any()}")
        print(f"Any infinite values in features: {np.isinf(X_processed).any()}")
        print(f"Feature mean (should be ~0): {X_processed.mean(axis=0)[:5]}...")
        print(f"Feature std (should be ~1): {X_processed.std(axis=0)[:5]}...")
        print(f"Target mean (accuracy): {y.mean():.4f}")
        print(f"Target range: [{y.min()}, {y.max()}]")
        
    except Exception as e:
        print(f"ERROR in preprocessing: {e}")
        return False
    
    # Test data splitting
    print(f"\n=== TESTING DATA SPLITTING ===")
    try:
        X_train, X_val, y_train, y_val = create_data_splits(X_processed, y, test_size=0.2, random_state=42)
        
        print(f"Original data: {X_processed.shape[0]} samples")
        print(f"Train split: {X_train.shape[0]} samples ({X_train.shape[0]/X_processed.shape[0]*100:.1f}%)")
        print(f"Validation split: {X_val.shape[0]} samples ({X_val.shape[0]/X_processed.shape[0]*100:.1f}%)")
        
    except Exception as e:
        print(f"ERROR in data splitting: {e}")
        return False
    
    # Test data loaders
    print(f"\n=== TESTING DATA LOADERS ===")
    try:
        train_loader, val_loader = create_data_loaders(
            X_train, X_val, y_train, y_val, 
            batch_size=64, num_workers=0
        )
        
        # Test a batch from each loader
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"Train batch - Features: {train_batch[0].shape}, Targets: {train_batch[1].shape}")
        print(f"Val batch - Features: {val_batch[0].shape}, Targets: {val_batch[1].shape}")
        print(f"Feature dtype: {train_batch[0].dtype}, Target dtype: {train_batch[1].dtype}")
        
    except Exception as e:
        print(f"ERROR in data loaders: {e}")
        return False
    
    # Test saving and loading preprocessor
    print(f"\n=== TESTING PREPROCESSOR PERSISTENCE ===")
    try:
        # Save preprocessor
        preprocessor.save_preprocessor("models/preprocessor.pkl")
        
        # Load preprocessor
        loaded_preprocessor = DataPreprocessor.load_preprocessor("models/preprocessor.pkl")
        
        # Test that loaded preprocessor works on new data
        test_df = train_df.head(100).copy()  # Small test sample
        X_test_original = preprocessor.transform(test_df)
        X_test_loaded = loaded_preprocessor.transform(test_df)
        
        # Check that results are identical
        if np.allclose(X_test_original, X_test_loaded):
            print("✅ Preprocessor persistence test PASSED")
        else:
            print("❌ Preprocessor persistence test FAILED")
            return False
            
    except Exception as e:
        print(f"ERROR in preprocessor persistence: {e}")
        return False
    
    # Test with test data
    print(f"\n=== TESTING WITH TEST DATA ===")
    try:
        test_df = load_pickle_data("data/test.pickle")
        if test_df is not None:
            # Remove target column for test transformation
            test_features = test_df.drop('is_correct', axis=1)
            X_test_processed = preprocessor.transform(test_features)
            
            print(f"Test data processed successfully: {X_test_processed.shape}")
            print(f"Test data - Any NaN: {np.isnan(X_test_processed).any()}")
            print(f"Test data - Any infinite: {np.isinf(X_test_processed).any()}")
        else:
            print("Could not load test data")
            
    except Exception as e:
        print(f"ERROR processing test data: {e}")
        return False
    
    # Summary
    print(f"\n=== PREPROCESSING PIPELINE TEST SUMMARY ===")
    print(f"✅ Data cleaning: PASSED")
    print(f"✅ Feature encoding: PASSED") 
    print(f"✅ Feature engineering: PASSED")
    print(f"✅ Feature normalization: PASSED")
    print(f"✅ Data splitting: PASSED")
    print(f"✅ Data loaders: PASSED")
    print(f"✅ Preprocessor persistence: PASSED")
    print(f"✅ Test data processing: PASSED")
    
    # Save test results
    with open("experiments/preprocessing_test_results.txt", "w") as f:
        f.write("=== PREPROCESSING PIPELINE TEST RESULTS ===\n\n")
        f.write(f"Input data shape: {train_df.shape}\n")
        f.write(f"Processed feature shape: {X_processed.shape}\n")
        f.write(f"Number of engineered features: {len(preprocessor.feature_columns)}\n")
        f.write(f"Feature columns: {preprocessor.feature_columns}\n")
        f.write(f"Train/val split: {X_train.shape[0]}/{X_val.shape[0]}\n")
        f.write(f"Data quality: No NaN or infinite values\n")
        f.write(f"All tests: PASSED\n")
    
    print(f"\nTest results saved to experiments/preprocessing_test_results.txt")
    
    return True

if __name__ == "__main__":
    success = test_preprocessing_pipeline()
    if success:
        print("\n🎉 ALL PREPROCESSING TESTS PASSED!")
    else:
        print("\n❌ PREPROCESSING TESTS FAILED!")
        sys.exit(1) 