"""
Tests for the molecular property prediction model.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import MolecularPropertyPredictor


class TestMolecularPropertyPredictor:
    """Test cases for MolecularPropertyPredictor model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        input_dim = 10
        hidden_dim = 64
        output_dim = 1
        
        model = MolecularPropertyPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        assert model.input_dim == input_dim
        assert model.hidden_dim == hidden_dim
        assert model.output_dim == output_dim
    
    def test_forward_pass(self):
        """Test forward pass with sample data."""
        batch_size = 4
        input_dim = 10
        output_dim = 1
        
        model = MolecularPropertyPredictor(input_dim=input_dim, output_dim=output_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
    
    def test_predict_method(self):
        """Test predict method."""
        input_dim = 10
        model = MolecularPropertyPredictor(input_dim=input_dim)
        
        x = torch.randn(2, input_dim)
        predictions = model.predict(x)
        
        assert predictions.shape == (2, 1)
    
    def test_model_training_mode(self):
        """Test that model can switch between train and eval modes."""
        model = MolecularPropertyPredictor(input_dim=10)
        
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training


class TestModelComponents:
    """Test individual model components."""
    
    def test_layer_dimensions(self):
        """Test that layer dimensions are set correctly."""
        input_dim = 20
        hidden_dim = 128
        output_dim = 3
        
        model = MolecularPropertyPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )
        
        assert model.fc1.in_features == input_dim
        assert model.fc1.out_features == hidden_dim
        assert model.fc2.in_features == hidden_dim
        assert model.fc2.out_features == hidden_dim
        assert model.fc3.in_features == hidden_dim
        assert model.fc3.out_features == output_dim


if __name__ == "__main__":
    pytest.main([__file__])
