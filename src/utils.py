"""
Utility functions for molecular property prediction.

This module contains helper functions for various tasks such as
metrics calculation, data processing, and visualization.
"""

import numpy as np
import torch


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        dict: Dictionary containing various metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_features(features, mean=None, std=None):
    """
    Normalize features using z-score normalization.
    
    Args:
        features (np.ndarray): Input features
        mean (np.ndarray, optional): Mean values for normalization
        std (np.ndarray, optional): Standard deviation values for normalization
        
    Returns:
        tuple: (normalized_features, mean, std)
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


def save_predictions(predictions, output_path, labels=None):
    """
    Save predictions to a file.
    
    Args:
        predictions (np.ndarray): Predicted values
        output_path (str): Path to save the predictions
        labels (np.ndarray, optional): True labels (if available)
    """
    if labels is not None:
        data = np.column_stack([labels, predictions])
        header = 'True,Predicted'
    else:
        data = predictions
        header = 'Predicted'
    
    np.savetxt(output_path, data, delimiter=',', header=header, comments='')
