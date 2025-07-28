"""
Data loading utilities for the neural network project.
Handles loading, exploring, and preprocessing data from pickle files.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_pickle_data(file_path):
    """
    Load data from pickle file.
    
    Args:
        file_path (str): Path to pickle file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {file_path}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def explore_data_structure(df, dataset_name="Dataset"):
    """
    Explore and print basic information about the dataset.
    
    Args:
        df (pandas.DataFrame): Dataset to explore
        dataset_name (str): Name for printing purposes
    """
    print(f"\n=== {dataset_name} Structure ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
    else:
        print("\nNo missing values found.")
    
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    
    return df


def analyze_target_distribution(df, target_col='is_correct'):
    """
    Analyze the distribution of the target variable.
    
    Args:
        df (pandas.DataFrame): Dataset
        target_col (str): Name of target column
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found in dataset")
        return
    
    print(f"\n=== Target Variable Analysis ({target_col}) ===")
    value_counts = df[target_col].value_counts()
    print(f"Value counts:\n{value_counts}")
    
    proportions = df[target_col].value_counts(normalize=True)
    print(f"\nProportions:\n{proportions}")
    
    # Create visualization
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    value_counts.plot(kind='bar')
    plt.title(f'{target_col} - Counts')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    proportions.plot(kind='pie', autopct='%1.1f%%')
    plt.title(f'{target_col} - Proportions')
    
    plt.tight_layout()
    plt.show()


def identify_features(df):
    """
    Identify and categorize different types of features.
    
    Args:
        df (pandas.DataFrame): Dataset
        
    Returns:
        dict: Dictionary with categorized features
    """
    features = {
        'numerical': [],
        'categorical': [],
        'binary': [],
        'potential_answer_features': []
    }
    
    print("\n=== Feature Analysis ===")
    
    for col in df.columns:
        if col == 'is_correct':  # Skip target variable
            continue
            
        unique_values = df[col].nunique()
        dtype = df[col].dtype
        
        print(f"\nColumn: {col}")
        print(f"  Data type: {dtype}")
        print(f"  Unique values: {unique_values}")
        print(f"  Sample values: {df[col].head(3).tolist()}")
        
        # Categorize features
        if unique_values == 2:
            features['binary'].append(col)
        elif dtype in ['int64', 'float64'] and unique_values > 10:
            features['numerical'].append(col)
        else:
            features['categorical'].append(col)
            
        # Look for potential answer features
        if any(keyword in col.lower() for keyword in ['answer', 'choice', 'selected', 'option']):
            features['potential_answer_features'].append(col)
    
    print("\n=== Feature Categorization ===")
    for category, cols in features.items():
        print(f"{category.title()}: {cols}")
    
    return features


def correlation_analysis(df, target_col='is_correct'):
    """
    Perform correlation analysis between features and target.
    
    Args:
        df (pandas.DataFrame): Dataset
        target_col (str): Target column name
    """
    if target_col not in df.columns:
        print(f"Target column '{target_col}' not found")
        return
    
    print(f"\n=== Correlation Analysis with {target_col} ===")
    
    # Select only numerical columns for correlation
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1 and target_col in numerical_cols:
        correlations = df[numerical_cols].corr()[target_col].sort_values(ascending=False)
        print("Correlations with target:")
        print(correlations)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        correlations.drop(target_col).plot(kind='barh')
        plt.title(f'Feature Correlations with {target_col}')
        plt.tight_layout()
        plt.show()
    else:
        print("Not enough numerical features for correlation analysis")


def comprehensive_eda(train_path, test_path):
    """
    Perform comprehensive exploratory data analysis.
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
    """
    print("Starting Comprehensive EDA...")
    
    # Load data
    train_df = load_pickle_data(train_path)
    test_df = load_pickle_data(test_path)
    
    if train_df is None or test_df is None:
        print("Failed to load data files")
        return None, None
    
    # Explore structure
    explore_data_structure(train_df, "Training Data")
    explore_data_structure(test_df, "Test Data")
    
    # Analyze target distribution (only in training data)
    analyze_target_distribution(train_df)
    
    # Identify features
    features = identify_features(train_df)
    
    # Correlation analysis
    correlation_analysis(train_df)
    
    return train_df, test_df, features


if __name__ == "__main__":
    # Run EDA if script is executed directly
    train_path = "data/train.pickle"
    test_path = "data/test.pickle"
    
    train_df, test_df, features = comprehensive_eda(train_path, test_path) 