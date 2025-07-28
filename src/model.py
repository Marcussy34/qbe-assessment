"""
Neural network models for answer correctness prediction.
Includes baseline and optimized architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BaselineNet(nn.Module):
    """
    Simple baseline neural network for answer correctness prediction.
    2-3 layer architecture with ReLU activation.
    """
    
    def __init__(self, input_dim=14, hidden_dim=64, dropout_rate=0.2):
        """
        Initialize baseline network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Hidden layer dimension
            dropout_rate (float): Dropout probability
        """
        super(BaselineNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.sigmoid(self.fc3(x))
        
        return x.squeeze(-1)  # Remove last dimension for binary classification


class OptimizedNet(nn.Module):
    """
    Optimized neural network with configurable architecture.
    Includes batch normalization, multiple hidden layers, and residual connections.
    """
    
    def __init__(self, input_dim=14, hidden_dims=[128, 64, 32], 
                 dropout_rate=0.3, use_batch_norm=True, use_residual=False):
        """
        Initialize optimized network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
            use_residual (bool): Whether to use residual connections
        """
        super(OptimizedNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build layers dynamically
        layers = []
        layer_dims = [input_dim] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())
        
        self.layers = nn.ModuleList(layers)
        
        # Store linear layers for residual connections
        self.linear_layers = [layer for layer in layers if isinstance(layer, nn.Linear)]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass with optional residual connections.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        residual = None
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                # Apply linear transformation
                if self.use_residual and residual is not None and x.size(-1) == residual.size(-1):
                    x = layer(x + residual)
                else:
                    x = layer(x)
                
                # Store for potential residual connection
                if layer != self.linear_layers[-1]:  # Not the output layer
                    residual = x
            else:
                # Apply non-linear layers (BatchNorm, ReLU, Dropout)
                x = layer(x)
        
        return x.squeeze(-1)  # Remove last dimension for binary classification


class DeepNet(nn.Module):
    """
    Deep neural network with many layers for complex pattern learning.
    """
    
    def __init__(self, input_dim=14, hidden_dims=[256, 128, 64, 32, 16], 
                 dropout_rate=0.3, use_batch_norm=True):
        """
        Initialize deep network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions
            dropout_rate (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(DeepNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        return self.network(x).squeeze(-1)


class WideNet(nn.Module):
    """
    Wide neural network with fewer layers but more neurons per layer.
    Good for capturing feature interactions.
    """
    
    def __init__(self, input_dim=14, hidden_dims=[512, 256], 
                 dropout_rate=0.4, use_batch_norm=True):
        """
        Initialize wide network.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer dimensions (should be wide)
            dropout_rate (float): Dropout probability
            use_batch_norm (bool): Whether to use batch normalization
        """
        super(WideNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout (higher for wide networks)
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        return self.network(x).squeeze(-1)


def create_model(model_type='baseline', input_dim=14, **kwargs):
    """
    Factory function to create different model types.
    
    Args:
        model_type (str): Type of model ('baseline', 'optimized', 'deep', 'wide')
        input_dim (int): Number of input features
        **kwargs: Additional arguments for specific models
        
    Returns:
        nn.Module: Neural network model
    """
    if model_type == 'baseline':
        return BaselineNet(input_dim=input_dim, **kwargs)
    elif model_type == 'optimized':
        return OptimizedNet(input_dim=input_dim, **kwargs)
    elif model_type == 'deep':
        return DeepNet(input_dim=input_dim, **kwargs)
    elif model_type == 'wide':
        return WideNet(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_dim=14):
    """
    Print a summary of the model architecture.
    
    Args:
        model (nn.Module): PyTorch model
        input_dim (int): Input dimension for test forward pass
    """
    print(f"=== MODEL SUMMARY ===")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, input_dim)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    
    print(f"Model architecture:")
    print(model)
    print("=" * 50) 