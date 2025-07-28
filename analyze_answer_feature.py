"""
Focused analysis to identify and understand the user's answer feature.
Based on initial EDA, feature1 appears to be the user's selected answer.
"""

import sys
sys.path.append('src')

import pandas as pd
from data_loader import load_pickle_data
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_answer_feature():
    """Analyze feature1 which likely contains the user's selected answer."""
    
    print("=== User Answer Feature Analysis ===\n")
    
    # Load training data
    train_df = load_pickle_data("data/train.pickle")
    
    if train_df is None:
        print("Failed to load training data")
        return
    
    # Focus on feature1 analysis
    print("=== Feature1 Analysis (Likely User's Answer) ===")
    print(f"Unique values in feature1: {sorted(train_df['feature1'].unique())}")
    print(f"Number of unique values: {train_df['feature1'].nunique()}")
    
    # Value counts
    feature1_counts = train_df['feature1'].value_counts()
    print(f"\nValue counts for feature1:")
    print(feature1_counts)
    
    # Analyze accuracy by answer choice
    print(f"\n=== Accuracy by Answer Choice ===")
    accuracy_by_choice = train_df.groupby('feature1')['is_correct'].agg(['count', 'sum', 'mean']).round(4)
    accuracy_by_choice.columns = ['total_answers', 'correct_answers', 'accuracy_rate']
    print(accuracy_by_choice)
    
    # Check if there's a pattern with correct answers
    print(f"\n=== Distribution Analysis ===")
    crosstab = pd.crosstab(train_df['feature1'], train_df['is_correct'], normalize='index')
    print("Proportions of correct/incorrect by answer choice:")
    print(crosstab.round(4))
    
    # Look at feature4 as well (might be number of choices)
    print(f"\n=== Feature4 Analysis (Possibly Number of Choices) ===")
    print(f"Unique values in feature4: {sorted(train_df['feature4'].unique())}")
    feature4_counts = train_df['feature4'].value_counts()
    print(f"Value counts for feature4:")
    print(feature4_counts)
    
    # Analyze relationship between feature1 and feature4
    print(f"\n=== Relationship between feature1 and feature4 ===")
    relationship = pd.crosstab(train_df['feature1'], train_df['feature4'])
    print("Cross-tabulation of feature1 vs feature4:")
    print(relationship)
    
    # Additional analysis for other categorical features
    print(f"\n=== Other Categorical Features ===")
    for col in ['user_id', 'question_id', 'batch_id']:
        print(f"{col}: {train_df[col].nunique()} unique values")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Accuracy by answer choice
    plt.subplot(2, 3, 1)
    accuracy_by_choice['accuracy_rate'].plot(kind='bar')
    plt.title('Accuracy Rate by Answer Choice (feature1)')
    plt.ylabel('Accuracy Rate')
    plt.xticks(rotation=0)
    
    # Plot 2: Distribution of answer choices
    plt.subplot(2, 3, 2)
    feature1_counts.plot(kind='bar')
    plt.title('Distribution of Answer Choices (feature1)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Plot 3: Feature4 distribution
    plt.subplot(2, 3, 3)
    feature4_counts.plot(kind='bar')
    plt.title('Distribution of feature4')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Plot 4: Heatmap of feature1 vs feature4
    plt.subplot(2, 3, 4)
    sns.heatmap(relationship, annot=True, fmt='d', cmap='Blues')
    plt.title('feature1 vs feature4 Relationship')
    
    # Plot 5: Accuracy by feature4
    plt.subplot(2, 3, 5)
    accuracy_by_feature4 = train_df.groupby('feature4')['is_correct'].mean()
    accuracy_by_feature4.plot(kind='bar')
    plt.title('Accuracy Rate by feature4')
    plt.ylabel('Accuracy Rate')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('experiments/answer_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed analysis
    with open("experiments/answer_feature_analysis.txt", "w") as f:
        f.write("=== User Answer Feature Analysis ===\n\n")
        f.write(f"Feature1 unique values: {sorted(train_df['feature1'].unique())}\n")
        f.write(f"Feature1 appears to be user's selected answer (a, b, c, d, etc.)\n\n")
        f.write("Accuracy by answer choice:\n")
        f.write(str(accuracy_by_choice))
        f.write("\n\nFeature4 analysis:\n")
        f.write(f"Unique values: {sorted(train_df['feature4'].unique())}\n")
        f.write("Feature4 likely represents number of answer choices in question\n")
        f.write("\nFeature1 vs Feature4 relationship:\n")
        f.write(str(relationship))
    
    print("\nAnalysis saved to experiments/answer_feature_analysis.txt")
    print("Visualization saved to experiments/answer_feature_analysis.png")
    
    return train_df

if __name__ == "__main__":
    train_df = analyze_answer_feature() 