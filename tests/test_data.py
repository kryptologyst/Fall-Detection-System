"""Unit tests for fall detection data module."""

import pytest
import numpy as np
from src.fall_detection.data import FallDetectionDataset, FallDetectionDataModule
from src.fall_detection.utils import set_seed


class TestFallDetectionDataset:
    """Test cases for FallDetectionDataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization with default parameters."""
        dataset = FallDetectionDataset()
        
        assert len(dataset) > 0
        assert dataset.data.shape[0] == len(dataset.labels)
        assert dataset.window_size == 100
        assert dataset.overlap == 0.5
        assert dataset.normalize is True
    
    def test_dataset_with_custom_params(self):
        """Test dataset initialization with custom parameters."""
        dataset = FallDetectionDataset(
            window_size=50,
            overlap=0.3,
            normalize=False,
            feature_extraction="statistical"
        )
        
        assert dataset.window_size == 50
        assert dataset.overlap == 0.3
        assert dataset.normalize is False
        assert dataset.feature_extraction == "statistical"
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        dataset = FallDetectionDataset()
        
        data, label = dataset[0]
        
        assert isinstance(data, np.ndarray)
        assert isinstance(label, (int, np.integer))
        assert label in [0, 1]
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        dataset = FallDetectionDataset()
        
        # Check data shape
        assert len(dataset.data.shape) >= 2
        
        # Check labels are binary
        unique_labels = np.unique(dataset.labels)
        assert set(unique_labels).issubset({0, 1})
        
        # Check we have both classes
        assert len(unique_labels) == 2
    
    def test_class_weights(self):
        """Test class weight calculation."""
        dataset = FallDetectionDataset()
        weights = dataset.get_class_weights()
        
        assert len(weights) == 2
        assert all(w > 0 for w in weights)
    
    def test_feature_extraction_types(self):
        """Test different feature extraction types."""
        feature_types = ["raw", "statistical", "frequency", "combined"]
        
        for feature_type in feature_types:
            dataset = FallDetectionDataset(feature_extraction=feature_type)
            assert dataset.feature_extraction == feature_type
            assert len(dataset) > 0


class TestFallDetectionDataModule:
    """Test cases for FallDetectionDataModule."""
    
    def test_data_module_initialization(self):
        """Test data module initialization."""
        data_module = FallDetectionDataModule(
            batch_size=16,
            test_size=0.3,
            val_size=0.2,
            random_state=123
        )
        
        assert data_module.batch_size == 16
        assert data_module.test_size == 0.3
        assert data_module.val_size == 0.2
        assert data_module.random_state == 123
    
    def test_data_module_setup(self):
        """Test data module setup."""
        data_module = FallDetectionDataModule()
        data_module.setup()
        
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert data_module.test_dataset is not None
        assert data_module.train_loader is not None
        assert data_module.val_loader is not None
        assert data_module.test_loader is not None
    
    def test_data_splits(self):
        """Test that data splits are properly created."""
        data_module = FallDetectionDataModule(test_size=0.2, val_size=0.1)
        data_module.setup()
        
        total_samples = (len(data_module.train_dataset) + 
                       len(data_module.val_dataset) + 
                       len(data_module.test_dataset))
        
        # Check that splits add up correctly
        assert len(data_module.train_dataset) > 0
        assert len(data_module.val_dataset) > 0
        assert len(data_module.test_dataset) > 0
        
        # Check that test split is approximately correct
        test_ratio = len(data_module.test_dataset) / total_samples
        assert abs(test_ratio - 0.2) < 0.1  # Allow some tolerance
    
    def test_data_loaders(self):
        """Test data loaders functionality."""
        data_module = FallDetectionDataModule(batch_size=8)
        data_module.setup()
        
        # Test train loader
        train_batch = next(iter(data_module.train_loader))
        assert len(train_batch) == 2  # data and labels
        assert train_batch[0].shape[0] <= 8  # batch size
        
        # Test val loader
        val_batch = next(iter(data_module.val_loader))
        assert len(val_batch) == 2
        
        # Test test loader
        test_batch = next(iter(data_module.test_loader))
        assert len(test_batch) == 2


if __name__ == "__main__":
    pytest.main([__file__])
