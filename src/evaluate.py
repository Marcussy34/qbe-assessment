"""
Evaluation utilities for neural network models.
Includes metrics calculation, model assessment, and result analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pickle


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics and visualizations.
    """
    
    def __init__(self, model, device):
        """
        Initialize evaluator.
        
        Args:
            model (nn.Module): Trained neural network model
            device (torch.device): Computation device
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, data_loader, return_probabilities=False):
        """
        Make predictions on a dataset.
        
        Args:
            data_loader (DataLoader): Data loader for predictions
            return_probabilities (bool): Whether to return probabilities or binary predictions
            
        Returns:
            tuple: (predictions, targets) as numpy arrays
        """
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                
                if return_probabilities:
                    predictions = output.cpu().numpy()
                else:
                    predictions = (output > 0.5).float().cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets)
    
    def calculate_metrics(self, data_loader):
        """
        Calculate comprehensive metrics for a dataset.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Get predictions and probabilities
        probabilities, targets = self.predict(data_loader, return_probabilities=True)
        predictions = (probabilities > 0.5).astype(int)
        targets = targets.astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, zero_division=0),
            'recall': recall_score(targets, predictions, zero_division=0),
            'f1_score': f1_score(targets, predictions, zero_division=0),
            'roc_auc': roc_auc_score(targets, probabilities),
            'confusion_matrix': confusion_matrix(targets, predictions),
            'classification_report': classification_report(targets, predictions, output_dict=True)
        }
        
        return metrics
    
    def evaluate_test_set(self, test_loader, save_path=None):
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_loader (DataLoader): Test data loader
            save_path (str): Path to save results (optional)
            
        Returns:
            dict: Complete evaluation results
        """
        print("=== TEST SET EVALUATION ===")
        
        # Calculate metrics
        metrics = self.calculate_metrics(test_loader)
        
        # Print results
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print("\nClassification Report:")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        print(report_df.round(4))
        
        # Save results if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(metrics, f)
            print(f"\nResults saved to {save_path}")
        
        return metrics
    
    def plot_confusion_matrix(self, test_loader, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            test_loader (DataLoader): Test data loader
            save_path (str): Path to save plot (optional)
        """
        metrics = self.calculate_metrics(test_loader)
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Incorrect', 'Correct'],
                   yticklabels=['Incorrect', 'Correct'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training history curves.
        
        Args:
            history (dict): Training history from trainer
            save_path (str): Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Training Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(history['train_acc'], label='Training Accuracy', color='blue')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], color='green')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # Validation accuracy zoom
        axes[1, 1].plot(history['val_acc'], color='red', linewidth=2)
        axes[1, 1].set_title('Validation Accuracy (Detailed)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def analyze_predictions(self, test_loader, save_path=None):
        """
        Analyze model predictions in detail.
        
        Args:
            test_loader (DataLoader): Test data loader
            save_path (str): Path to save analysis (optional)
            
        Returns:
            dict: Prediction analysis results
        """
        probabilities, targets = self.predict(test_loader, return_probabilities=True)
        predictions = (probabilities > 0.5).astype(int)
        
        # Prediction confidence analysis
        confidence = np.abs(probabilities - 0.5)  # Distance from decision boundary
        
        analysis = {
            'total_samples': len(targets),
            'correct_predictions': np.sum(predictions == targets),
            'accuracy': np.mean(predictions == targets),
            'avg_confidence': np.mean(confidence),
            'high_confidence_correct': np.sum((confidence > 0.4) & (predictions == targets)),
            'high_confidence_incorrect': np.sum((confidence > 0.4) & (predictions != targets)),
            'low_confidence_samples': np.sum(confidence < 0.1),
        }
        
        print("=== PREDICTION ANALYSIS ===")
        print(f"Total samples: {analysis['total_samples']}")
        print(f"Correct predictions: {analysis['correct_predictions']}")
        print(f"Accuracy: {analysis['accuracy']:.4f} ({analysis['accuracy']*100:.2f}%)")
        print(f"Average confidence: {analysis['avg_confidence']:.4f}")
        print(f"High confidence correct: {analysis['high_confidence_correct']}")
        print(f"High confidence incorrect: {analysis['high_confidence_incorrect']}")
        print(f"Low confidence samples: {analysis['low_confidence_samples']}")
        
        # Plot confidence distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(probabilities, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(confidence, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence (Distance from 0.5)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path.replace('.pkl', '_confidence.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(analysis, f)
            print(f"Analysis saved to {save_path}")
        
        return analysis


def compare_models(models_dict, test_loader, device):
    """
    Compare multiple models on the same test set.
    
    Args:
        models_dict (dict): Dictionary of {model_name: model} pairs
        test_loader (DataLoader): Test data loader
        device (torch.device): Computation device
        
    Returns:
        pd.DataFrame: Comparison results
    """
    results = []
    
    for model_name, model in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        evaluator = ModelEvaluator(model, device)
        metrics = evaluator.calculate_metrics(test_loader)
        
        results.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'ROC AUC': metrics['roc_auc']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.round(4)
    
    print("\n=== MODEL COMPARISON ===")
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(models_dict))
    width = 0.15
    
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(x + i * width, comparison_df[metric], width, label=metric)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width * 2, comparison_df['Model'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return comparison_df


def final_test_evaluation(model, test_loader, device, preprocessor_path=None):
    """
    Perform final evaluation on test set for submission.
    
    Args:
        model (nn.Module): Best trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Computation device
        preprocessor_path (str): Path to preprocessor for documentation
        
    Returns:
        dict: Final test results
    """
    print("=== FINAL TEST SET EVALUATION ===")
    
    evaluator = ModelEvaluator(model, device)
    
    # Calculate all metrics
    final_results = evaluator.evaluate_test_set(test_loader)
    
    # Create visualizations
    evaluator.plot_confusion_matrix(test_loader, 'experiments/final_confusion_matrix.png')
    
    # Detailed analysis
    analysis = evaluator.analyze_predictions(test_loader, 'experiments/final_prediction_analysis.pkl')
    
    # Summary for reporting
    final_accuracy = final_results['accuracy']
    print(f"\n{'='*50}")
    print(f"FINAL TEST ACCURACY: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"{'='*50}")
    
    # Check if target is met
    if final_accuracy >= 0.95:
        print("🎉 TARGET ACHIEVED: 95%+ accuracy!")
    else:
        print(f"❌ Target not met. Need {0.95 - final_accuracy:.4f} more accuracy.")
    
    return final_results 