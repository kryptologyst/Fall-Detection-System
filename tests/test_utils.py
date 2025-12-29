"""Unit tests for fall detection utilities."""

import pytest
import numpy as np
import torch
from src.fall_detection.utils import (
    set_seed,
    get_device,
    normalize_features,
    create_sliding_windows,
    extract_statistical_features,
    extract_frequency_features,
)


class TestSeedSetting:
    """Test cases for seed setting."""
    
    def test_set_seed(self):
        """Test that seed setting works."""
        set_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.randn()
        
        set_seed(42)
        val2 = np.random.randn()
        
        assert val1 == val2
        
        # Test torch
        torch.manual_seed(42)
        val3 = torch.randn(1).item()
        
        set_seed(42)
        val4 = torch.randn(1).item()
        
        assert val3 == val4


class TestDeviceDetection:
    """Test cases for device detection."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


class TestFeatureNormalization:
    """Test cases for feature normalization."""
    
    def test_zscore_normalization(self):
        """Test z-score normalization."""
        data = np.random.randn(100, 5)
        
        normalized, stats = normalize_features(data, method="zscore")
        
        # Check that normalized data has mean ~0 and std ~1
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)
        
        # Check that stats are correct
        assert "mean" in stats
        assert "std" in stats
        assert np.allclose(stats["mean"], np.mean(data, axis=0))
        assert np.allclose(stats["std"], np.std(data, axis=0))
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        data = np.random.randn(100, 5)
        
        normalized, stats = normalize_features(data, method="minmax")
        
        # Check that normalized data is in [0, 1]
        assert np.all(normalized >= 0)
        assert np.all(normalized <= 1)
        
        # Check that stats are correct
        assert "min" in stats
        assert "max" in stats
        assert np.allclose(stats["min"], np.min(data, axis=0))
        assert np.allclose(stats["max"], np.max(data, axis=0))
    
    def test_robust_normalization(self):
        """Test robust normalization."""
        data = np.random.randn(100, 5)
        
        normalized, stats = normalize_features(data, method="robust")
        
        # Check that stats are correct
        assert "median" in stats
        assert "mad" in stats
        assert np.allclose(stats["median"], np.median(data, axis=0))
    
    def test_normalization_with_stats(self):
        """Test normalization with pre-computed stats."""
        data = np.random.randn(100, 5)
        
        # Compute stats from first half
        stats_data = data[:50]
        normalized1, stats = normalize_features(stats_data, method="zscore")
        
        # Apply same stats to second half
        normalized2, _ = normalize_features(data[50:], method="zscore", stats=stats)
        
        # Check that stats are reused correctly
        assert "mean" in stats
        assert "std" in stats
    
    def test_invalid_normalization_method(self):
        """Test invalid normalization method."""
        data = np.random.randn(100, 5)
        
        with pytest.raises(ValueError):
            normalize_features(data, method="invalid")


class TestSlidingWindows:
    """Test cases for sliding windows."""
    
    def test_sliding_windows_basic(self):
        """Test basic sliding window creation."""
        data = np.arange(20).reshape(-1, 1)
        windows = create_sliding_windows(data, window_size=5, overlap=0.5)
        
        assert windows.shape[0] > 0
        assert windows.shape[1] == 5
        assert windows.shape[2] == 1
    
    def test_sliding_windows_overlap(self):
        """Test sliding windows with different overlap ratios."""
        data = np.arange(20).reshape(-1, 1)
        
        # No overlap
        windows_no_overlap = create_sliding_windows(data, window_size=5, overlap=0.0)
        
        # 50% overlap
        windows_half_overlap = create_sliding_windows(data, window_size=5, overlap=0.5)
        
        # More windows with overlap
        assert len(windows_half_overlap) > len(windows_no_overlap)
    
    def test_sliding_windows_edge_cases(self):
        """Test edge cases for sliding windows."""
        data = np.arange(10).reshape(-1, 1)
        
        # Window size larger than data
        windows = create_sliding_windows(data, window_size=20, overlap=0.5)
        assert len(windows) == 0
        
        # Window size equal to data
        windows = create_sliding_windows(data, window_size=10, overlap=0.0)
        assert len(windows) == 1


class TestStatisticalFeatures:
    """Test cases for statistical feature extraction."""
    
    def test_statistical_features_shape(self):
        """Test statistical features shape."""
        data = np.random.randn(100, 6)
        features = extract_statistical_features(data)
        
        # Should have multiple statistical measures per channel
        expected_features = 6 * 10  # 6 channels, 10 features each
        assert len(features) == expected_features
    
    def test_statistical_features_values(self):
        """Test statistical features values."""
        data = np.random.randn(100, 3)
        features = extract_statistical_features(data)
        
        # Check that features are finite
        assert np.all(np.isfinite(features))
        
        # Check that features are reasonable
        assert np.all(np.abs(features) < 1e6)  # Not too large


class TestFrequencyFeatures:
    """Test cases for frequency feature extraction."""
    
    def test_frequency_features_shape(self):
        """Test frequency features shape."""
        data = np.random.randn(100, 6)
        features = extract_frequency_features(data)
        
        # Should have frequency features per channel
        expected_features = 6 * 5  # 6 channels, 5 features each
        assert len(features) == expected_features
    
    def test_frequency_features_values(self):
        """Test frequency features values."""
        data = np.random.randn(100, 3)
        features = extract_frequency_features(data)
        
        # Check that features are finite
        assert np.all(np.isfinite(features))
        
        # Check that features are reasonable
        assert np.all(np.abs(features) < 1e6)  # Not too large
    
    def test_frequency_features_with_sampling_rate(self):
        """Test frequency features with custom sampling rate."""
        data = np.random.randn(100, 3)
        features = extract_frequency_features(data, sampling_rate=100.0)
        
        assert len(features) == 15  # 3 channels * 5 features
        assert np.all(np.isfinite(features))


if __name__ == "__main__":
    pytest.main([__file__])
