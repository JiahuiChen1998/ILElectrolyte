"""
Example script for making predictions with a trained model.

This script demonstrates how to:
1. Load a trained model
2. Load test data
3. Make predictions
4. Evaluate results
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import MolecularPropertyPredictor
from predict import Predictor
from utils import calculate_metrics


def main():
    # Configuration
    input_dim = 100
    hidden_dim = 128
    output_dim = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Initialize model architecture
    model = MolecularPropertyPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    # Load trained model
    checkpoint_path = 'models/best_model.pth'
    try:
        predictor = Predictor.from_checkpoint(checkpoint_path, model, device)
        print(f"Model loaded from {checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {checkpoint_path}")
        print("Please train a model first using train_example.py")
        return
    
    # Generate dummy test data for demonstration
    # Replace this with actual data loading
    n_test_samples = 100
    test_features = np.random.randn(n_test_samples, input_dim)
    test_labels = np.random.randn(n_test_samples, output_dim)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict(test_features)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"First 5 predictions: {predictions[:5].flatten()}")
    
    # Calculate metrics if true labels are available
    if test_labels is not None:
        metrics = calculate_metrics(test_labels, predictions)
        
        print("\nEvaluation Metrics:")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    # Save predictions
    output_path = 'outputs/predictions.csv'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if test_labels is not None:
        data = np.column_stack([test_labels, predictions])
        header = 'True,Predicted'
    else:
        data = predictions
        header = 'Predicted'
    
    np.savetxt(output_path, data, delimiter=',', header=header, comments='')
    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
