"""
Example script for training a molecular property prediction model.

This script demonstrates how to:
1. Load and preprocess data
2. Create a model
3. Train the model
4. Save the trained model
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import MolecularPropertyPredictor
from data_loader import create_data_loader
from train import Trainer
from utils import set_seed


def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    input_dim = 100
    hidden_dim = 128
    output_dim = 1
    batch_size = 32
    epochs = 50
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Generate dummy data for demonstration
    # Replace this with actual data loading
    n_samples = 1000
    train_features = np.random.randn(n_samples, input_dim)
    train_labels = np.random.randn(n_samples, output_dim)
    
    val_features = np.random.randn(200, input_dim)
    val_labels = np.random.randn(200, output_dim)
    
    # Create data loaders
    train_loader = create_data_loader(
        train_features, 
        train_labels, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = create_data_loader(
        val_features, 
        val_labels, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # Initialize model
    model = MolecularPropertyPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = Trainer(model, device=device, learning_rate=learning_rate)
    
    # Train the model
    print("\nStarting training...")
    history = trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        save_path='models/best_model.pth'
    )
    
    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    
    # Save final model
    trainer.save_model('models/final_model.pth')
    print("\nModel saved to 'models/final_model.pth'")


if __name__ == "__main__":
    main()
