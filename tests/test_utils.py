"""
Tests for utility functions.
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import calculate_metrics, set_seed, count_parameters, normalize_features
from model import MolecularPropertyPredictor


class TestMetrics:
    """Test cases for metric calculations."""
    
    def test_calculate_metrics(self):
        """Test metric calculation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        assert metrics['mse'] > 0
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
        assert 0 <= metrics['r2'] <= 1
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert metrics['mse'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mae'] == 0
        assert metrics['r2'] == 1.0


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test that seed setting works."""
        set_seed(42)
        rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()
        
        set_seed(42)
        rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()
        
        assert rand1 == rand2
        assert torch_rand1 == torch_rand2
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = MolecularPropertyPredictor(input_dim=10, hidden_dim=64, output_dim=1)
        
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        normalized, mean, std = normalize_features(features)
        
        assert normalized.shape == features.shape
        assert mean.shape == (2,)
        assert std.shape == (2,)
        
        # Check that normalized features have mean ~0 and std ~1
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
