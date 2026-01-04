"""
Tests for data loading and preprocessing utilities.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import MolecularDataset, create_data_loader, preprocess_features


class TestMolecularDataset:
    """Test cases for MolecularDataset class."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization with numpy arrays."""
        features = np.random.randn(10, 5)
        labels = np.random.randn(10, 1)
        
        dataset = MolecularDataset(features, labels)
        
        assert len(dataset) == 10
        assert isinstance(dataset.features, torch.Tensor)
        assert isinstance(dataset.labels, torch.Tensor)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        features = np.random.randn(10, 5)
        labels = np.random.randn(10, 1)
        
        dataset = MolecularDataset(features, labels)
        
        feature, label = dataset[0]
        assert feature.shape == (5,)
        assert label.shape == (1,)
    
    def test_dataset_without_labels(self):
        """Test dataset with features only."""
        features = np.random.randn(10, 5)
        
        dataset = MolecularDataset(features)
        
        assert len(dataset) == 10
        assert dataset.labels is None
        
        feature = dataset[0]
        assert feature.shape == (5,)


class TestDataLoader:
    """Test cases for data loader creation."""
    
    def test_create_data_loader(self):
        """Test data loader creation."""
        features = np.random.randn(20, 5)
        labels = np.random.randn(20, 1)
        
        loader = create_data_loader(features, labels, batch_size=4, shuffle=False)
        
        batch_count = 0
        for batch_features, batch_labels in loader:
            batch_count += 1
            assert batch_features.shape[0] <= 4
            assert batch_labels.shape[0] <= 4
        
        assert batch_count == 5  # 20 samples / 4 batch_size


if __name__ == "__main__":
    pytest.main([__file__])
