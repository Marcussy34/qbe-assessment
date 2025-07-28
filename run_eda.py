"""
Script to run comprehensive exploratory data analysis.
This will help us understand the structure and characteristics of our data.
"""

import sys
import os
sys.path.append('src')

from utils import set_random_seeds, print_system_info
from data_loader import comprehensive_eda

def main():
    """Run the complete exploratory data analysis."""
    print("=== Neural Network Assessment - Data Exploration ===\n")
    
    # Set up reproducibility
    set_random_seeds(42)
    print_system_info()
    
    # Run comprehensive EDA
    train_path = "data/train.pickle"
    test_path = "data/test.pickle"
    
    print("\n" + "="*60)
    print("STARTING EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    try:
        train_df, test_df, features = comprehensive_eda(train_path, test_path)
        
        if train_df is not None and test_df is not None:
            print("\n" + "="*60)
            print("EDA COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Save key findings to a file
            with open("experiments/eda_summary.txt", "w") as f:
                f.write("=== EDA Summary ===\n")
                f.write(f"Training data shape: {train_df.shape}\n")
                f.write(f"Test data shape: {test_df.shape}\n")
                f.write(f"Features categorization: {features}\n")
                f.write(f"Target distribution: {train_df['is_correct'].value_counts().to_dict()}\n")
            
            print("Summary saved to experiments/eda_summary.txt")
            
        else:
            print("ERROR: Failed to load data files!")
            
    except Exception as e:
        print(f"ERROR during EDA: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 