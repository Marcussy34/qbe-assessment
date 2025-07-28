"""
Training utilities for neural network models.
Includes training loops, loss functions, optimizers, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import numpy as np
import time
from collections import defaultdict
import pickle
from pathlib import Path


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        """
        Check if training should stop.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to potentially save weights
            
        Returns:
            bool: Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
        
        return self.early_stop


class ModelTrainer:
    """
    Comprehensive model trainer with configurable training strategy.
    """
    
    def __init__(self, model, device, train_loader, val_loader, 
                 learning_rate=0.001, optimizer_type='adam', 
                 scheduler_type='plateau', early_stopping_patience=10):
        """
        Initialize trainer.
        
        Args:
            model (nn.Module): Neural network model
            device (torch.device): Training device
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            learning_rate (float): Initial learning rate
            optimizer_type (str): Optimizer type ('adam', 'sgd', 'adamw')
            scheduler_type (str): Scheduler type ('plateau', 'step', 'cosine', 'none')
            early_stopping_patience (int): Early stopping patience
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function for binary classification
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer(optimizer_type, learning_rate)
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler(scheduler_type)
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Training history
        self.history = defaultdict(list)
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Early stopping patience: {early_stopping_patience}")
    
    def _create_optimizer(self, optimizer_type, learning_rate):
        """Create optimizer based on type."""
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        elif optimizer_type.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, 
                           momentum=0.9, weight_decay=1e-5)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type):
        """Create learning rate scheduler based on type."""
        if scheduler_type.lower() == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, 
                                   patience=5)
        elif scheduler_type.lower() == 'step':
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_type.lower() == 'cosine':
            return CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-6)
        elif scheduler_type.lower() == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs=50, print_every=5, save_path=None):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs (int): Number of training epochs
            print_every (int): Print progress every N epochs
            save_path (str): Path to save best model (optional)
            
        Returns:
            dict: Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Save best model if path provided
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"Best model saved to {save_path}")
        
        return dict(self.history)
    
    def save_history(self, filepath):
        """Save training history to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.history), f)
        print(f"Training history saved to {filepath}")
    
    def load_history(self, filepath):
        """Load training history from file."""
        with open(filepath, 'rb') as f:
            self.history = defaultdict(list, pickle.load(f))
        print(f"Training history loaded from {filepath}")


def train_model_with_config(model, train_loader, val_loader, device, config):
    """
    Train a model with a specific configuration.
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Training device
        config (dict): Training configuration
        
    Returns:
        tuple: (trained_model, training_history, best_val_acc)
    """
    trainer = ModelTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config.get('learning_rate', 0.001),
        optimizer_type=config.get('optimizer', 'adam'),
        scheduler_type=config.get('scheduler', 'plateau'),
        early_stopping_patience=config.get('patience', 10)
    )
    
    # Train the model
    history = trainer.train(
        num_epochs=config.get('epochs', 50),
        print_every=config.get('print_every', 5),
        save_path=config.get('save_path', None)
    )
    
    # Get best validation accuracy
    best_val_acc = max(history['val_acc'])
    
    print(f"Best validation accuracy: {best_val_acc:.4f}%")
    
    return trainer.model, history, best_val_acc


def get_default_configs():
    """
    Get dictionary of default training configurations for different experiments.
    
    Returns:
        dict: Dictionary of training configurations
    """
    configs = {
        'baseline': {
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'epochs': 50,
            'patience': 10,
            'print_every': 5
        },
        'high_lr': {
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'epochs': 50,
            'patience': 10,
            'print_every': 5
        },
        'low_lr': {
            'learning_rate': 0.0001,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'epochs': 100,
            'patience': 15,
            'print_every': 10
        },
        'sgd_momentum': {
            'learning_rate': 0.01,
            'optimizer': 'sgd',
            'scheduler': 'step',
            'epochs': 50,
            'patience': 10,
            'print_every': 5
        },
        'adamw': {
            'learning_rate': 0.001,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'epochs': 50,
            'patience': 10,
            'print_every': 5
        }
    }
    
    return configs 