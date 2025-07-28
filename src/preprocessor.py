"""
Data preprocessing pipeline for the neural network project.
Handles feature encoding, normalization, and data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path


class AnswerDataset(Dataset):
    """PyTorch dataset for answer correctness prediction."""
    
    def __init__(self, features, targets):
        """
        Initialize dataset.
        
        Args:
            features (np.ndarray): Feature matrix
            targets (np.ndarray): Target vector
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline.
    Handles cleaning, encoding, normalization, and feature engineering.
    """
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.feature1_encoder = LabelEncoder()  # For user's answer choice
        self.numerical_scaler = StandardScaler()  # For numerical features
        self.feature_columns = None
        self.is_fitted = False
        
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        df_clean = df.copy()
        
        # Handle missing values in feature5 (fill with median)
        if df_clean['feature5'].isnull().sum() > 0:
            median_value = df_clean['feature5'].median()
            df_clean.loc[:, 'feature5'] = df_clean['feature5'].fillna(median_value)
            print(f"Filled {df['feature5'].isnull().sum()} missing values in feature5 with median: {median_value:.4f}")
        
        return df_clean
    
    def encode_categorical_features(self, df, fit=True):
        """
        Encode categorical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            fit (bool): Whether to fit encoders (True for training, False for test)
            
        Returns:
            pd.DataFrame: Dataframe with encoded features
        """
        df_encoded = df.copy()
        
        # Encode feature1 (user's answer choice)
        if fit:
            df_encoded['feature1_encoded'] = self.feature1_encoder.fit_transform(df_encoded['feature1'])
        else:
            df_encoded['feature1_encoded'] = self.feature1_encoder.transform(df_encoded['feature1'])
        
        # feature4 is already numerical (number of choices), keep as is
        df_encoded['feature4_normalized'] = df_encoded['feature4']
        
        print(f"Encoded feature1: {len(self.feature1_encoder.classes_)} unique values")
        print(f"Feature1 mapping: {dict(zip(self.feature1_encoder.classes_, range(len(self.feature1_encoder.classes_))))}")
        
        return df_encoded
    
    def create_engineered_features(self, df):
        """
        Create engineered features based on domain knowledge.
        
        Args:
            df (pd.DataFrame): Input dataframe with encoded features
            
        Returns:
            pd.DataFrame: Dataframe with additional engineered features
        """
        df_engineered = df.copy()
        
        # Feature engineering based on answer choice patterns
        # 1. Answer position relative to number of choices (normalized)
        # Handle division by zero for single-choice questions
        denominator = df_engineered['feature4'] - 1
        df_engineered['answer_position_ratio'] = np.where(
            denominator == 0, 
            0.0,  # For single choice questions, set ratio to 0
            df_engineered['feature1_encoded'] / denominator
        )
        
        # 2. Binary features for common answer positions
        df_engineered['is_first_choice'] = (df_engineered['feature1_encoded'] == 0).astype(int)
        df_engineered['is_last_choice'] = (df_engineered['feature1_encoded'] == (df_engineered['feature4'] - 1)).astype(int)
        df_engineered['is_middle_choice'] = ((df_engineered['feature1_encoded'] > 0) & 
                                           (df_engineered['feature1_encoded'] < df_engineered['feature4'] - 1)).astype(int)
        
        # 3. Interaction between numerical features and answer choice
        df_engineered['feature2_answer_interaction'] = df_engineered['feature2'] * df_engineered['feature1_encoded']
        df_engineered['feature3_answer_interaction'] = df_engineered['feature3'] * df_engineered['feature1_encoded']
        df_engineered['feature5_answer_interaction'] = df_engineered['feature5'] * df_engineered['feature1_encoded']
        
        # 4. Question complexity indicators
        df_engineered['has_many_choices'] = (df_engineered['feature4'] >= 5).astype(int)
        df_engineered['is_binary_question'] = (df_engineered['feature4'] == 2).astype(int)
        
        print(f"Created {len(df_engineered.columns) - len(df.columns)} engineered features")
        
        return df_engineered
    
    def select_features_for_model(self, df):
        """
        Select and prepare final feature set for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe with all features
            
        Returns:
            pd.DataFrame: Dataframe with selected features
        """
        # Core features for the model
        feature_columns = [
            # Primary answer choice feature
            'feature1_encoded',
            
            # Context features
            'feature4_normalized',  # number of choices
            
            # Numerical features
            'feature2', 'feature3', 'feature5',
            
            # Engineered features
            'answer_position_ratio',
            'is_first_choice', 'is_last_choice', 'is_middle_choice',
            'feature2_answer_interaction', 'feature3_answer_interaction', 'feature5_answer_interaction',
            'has_many_choices', 'is_binary_question'
        ]
        
        self.feature_columns = feature_columns
        return df[feature_columns]
    
    def normalize_features(self, X, fit=True):
        """
        Normalize numerical features.
        
        Args:
            X (pd.DataFrame or np.ndarray): Feature matrix
            fit (bool): Whether to fit scaler (True for training, False for test)
            
        Returns:
            np.ndarray: Normalized feature matrix
        """
        if fit:
            X_normalized = self.numerical_scaler.fit_transform(X)
        else:
            X_normalized = self.numerical_scaler.transform(X)
        
        print(f"Normalized features to mean=0, std=1")
        return X_normalized
    
    def fit_transform(self, df, target_col='is_correct'):
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            df (pd.DataFrame): Training dataframe
            target_col (str): Target column name
            
        Returns:
            tuple: (X_processed, y) where X is features and y is targets
        """
        print("=== FITTING PREPROCESSING PIPELINE ===")
        
        # Step 1: Clean data
        df_clean = self.clean_data(df)
        
        # Step 2: Encode categorical features
        df_encoded = self.encode_categorical_features(df_clean, fit=True)
        
        # Step 3: Create engineered features
        df_engineered = self.create_engineered_features(df_encoded)
        
        # Step 4: Select features for model
        X_selected = self.select_features_for_model(df_engineered)
        
        # Step 5: Normalize features
        X_processed = self.normalize_features(X_selected, fit=True)
        
        # Extract target
        y = df_clean[target_col].astype(float).values
        
        self.is_fitted = True
        print(f"Preprocessing complete: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        return X_processed, y
    
    def transform(self, df):
        """
        Apply fitted preprocessing pipeline to new data.
        
        Args:
            df (pd.DataFrame): New dataframe to transform
            
        Returns:
            np.ndarray: Processed feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        print("=== APPLYING PREPROCESSING PIPELINE ===")
        
        # Apply same steps as fit_transform but without fitting
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical_features(df_clean, fit=False)
        df_engineered = self.create_engineered_features(df_encoded)
        X_selected = self.select_features_for_model(df_engineered)
        X_processed = self.normalize_features(X_selected, fit=False)
        
        print(f"Transformation complete: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
        
        return X_processed
    
    def save_preprocessor(self, filepath):
        """Save the fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load_preprocessor(cls, filepath):
        """Load a fitted preprocessor."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def create_data_splits(X, y, test_size=0.2, random_state=42):
    """
    Create stratified train/validation splits.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Proportion for validation set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data splits created:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Train accuracy: {y_train.mean():.4f}")
    print(f"  Val accuracy: {y_val.mean():.4f}")
    
    return X_train, X_val, y_train, y_val


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size=64, num_workers=0):
    """
    Create PyTorch data loaders for training and validation.
    
    Args:
        X_train, X_val: Feature matrices
        y_train, y_val: Target vectors
        batch_size (int): Batch size for training
        num_workers (int): Number of worker processes
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = AnswerDataset(X_train, y_train)
    val_dataset = AnswerDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader 