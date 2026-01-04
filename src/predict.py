"""
Prediction and inference utilities.

This module provides functions for making predictions using trained models
on new molecular data.
"""

import torch
import numpy as np


class Predictor:
    """
    Predictor class for making predictions with trained models.
    """
    
    def __init__(self, model, device='cpu'):
        """
        Initialize the predictor.
        
        Args:
            model (nn.Module): Trained model
            device (str): Device to use for prediction ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(self, features):
        """
        Make predictions on input features.
        
        Args:
            features (np.ndarray or torch.Tensor): Input molecular features
            
        Returns:
            np.ndarray: Predicted property values
        """
        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        features = features.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)
        
        return predictions.cpu().numpy()
    
    def predict_batch(self, data_loader):
        """
        Make predictions on batches of data.
        
        Args:
            data_loader (DataLoader): Data loader with input features
            
        Returns:
            np.ndarray: All predictions concatenated
        """
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Handle both (features, labels) and features only
                if isinstance(batch, (list, tuple)):
                    features = batch[0]
                else:
                    features = batch
                
                features = features.to(self.device)
                predictions = self.model(features)
                all_predictions.append(predictions.cpu().numpy())
        
        return np.concatenate(all_predictions, axis=0)
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, device='cpu'):
        """
        Create a predictor from a saved checkpoint.
        
        Args:
            checkpoint_path (str): Path to the model checkpoint
            model (nn.Module): Model architecture (will load weights)
            device (str): Device to use for prediction
            
        Returns:
            Predictor: Initialized predictor with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return cls(model, device)
