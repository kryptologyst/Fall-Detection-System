"""Unit tests for fall detection models."""

import pytest
import numpy as np
import torch
from src.fall_detection.models import (
    RandomForestFallDetector,
    CNN1DFallDetector,
    LSTMFallDetector,
    TransformerFallDetector,
)
from src.fall_detection.data import FallDetectionDataset
from src.fall_detection.utils import set_seed


class TestRandomForestFallDetector:
    """Test cases for RandomForestFallDetector."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = RandomForestFallDetector(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        
        assert model.n_estimators == 50
        assert model.max_depth == 10
        assert model.random_state == 42
        assert not model.is_fitted
    
    def test_model_fit_predict(self):
        """Test model training and prediction."""
        # Create synthetic data
        dataset = FallDetectionDataset(feature_extraction="statistical")
        
        model = RandomForestFallDetector(random_state=42)
        model.fit(dataset.data, dataset.labels)
        
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(dataset.data[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability prediction
        probabilities = model.predict_proba(dataset.data[:10])
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_feature_importance(self):
        """Test feature importance calculation."""
        dataset = FallDetectionDataset(feature_extraction="statistical")
        
        model = RandomForestFallDetector(random_state=42)
        model.fit(dataset.data, dataset.labels)
        
        importance = model.get_feature_importance()
        assert len(importance) == dataset.data.shape[1]
        assert all(imp >= 0 for imp in importance)


class TestCNN1DFallDetector:
    """Test cases for CNN1DFallDetector."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = CNN1DFallDetector(
            input_channels=6,
            sequence_length=100,
            hidden_dim=32,
            num_layers=2
        )
        
        assert model.input_channels == 6
        assert model.sequence_length == 100
        assert model.hidden_dim == 32
        assert model.num_layers == 2
        assert not model.is_fitted
    
    def test_model_fit_predict(self):
        """Test model training and prediction."""
        # Create synthetic data
        dataset = FallDetectionDataset(feature_extraction="raw", window_size=100)
        
        model = CNN1DFallDetector(
            input_channels=6,
            sequence_length=100,
            hidden_dim=32,
            num_layers=2
        )
        
        # Train for a few epochs
        model.fit(dataset.data[:100], dataset.labels[:100], epochs=5, batch_size=16)
        
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(dataset.data[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test probability prediction
        probabilities = model.predict_proba(dataset.data[:10])
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestLSTMFallDetector:
    """Test cases for LSTMFallDetector."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = LSTMFallDetector(
            input_size=6,
            hidden_size=32,
            num_layers=2,
            bidirectional=True
        )
        
        assert model.input_size == 6
        assert model.hidden_size == 32
        assert model.num_layers == 2
        assert model.bidirectional is True
        assert not model.is_fitted
    
    def test_model_fit_predict(self):
        """Test model training and prediction."""
        # Create synthetic data
        dataset = FallDetectionDataset(feature_extraction="raw", window_size=100)
        
        model = LSTMFallDetector(
            input_size=6,
            hidden_size=32,
            num_layers=2
        )
        
        # Train for a few epochs
        model.fit(dataset.data[:100], dataset.labels[:100], epochs=5, batch_size=16)
        
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(dataset.data[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


class TestTransformerFallDetector:
    """Test cases for TransformerFallDetector."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = TransformerFallDetector(
            input_size=6,
            d_model=32,
            nhead=2,
            num_layers=2
        )
        
        assert model.input_size == 6
        assert model.d_model == 32
        assert model.nhead == 2
        assert model.num_layers == 2
        assert not model.is_fitted
    
    def test_model_fit_predict(self):
        """Test model training and prediction."""
        # Create synthetic data
        dataset = FallDetectionDataset(feature_extraction="raw", window_size=100)
        
        model = TransformerFallDetector(
            input_size=6,
            d_model=32,
            nhead=2,
            num_layers=2
        )
        
        # Train for a few epochs
        model.fit(dataset.data[:100], dataset.labels[:100], epochs=5, batch_size=16)
        
        assert model.is_fitted
        
        # Test prediction
        predictions = model.predict(dataset.data[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)


class TestModelConsistency:
    """Test model consistency and edge cases."""
    
    def test_all_models_same_input(self):
        """Test that all models can handle the same input format."""
        dataset = FallDetectionDataset(feature_extraction="raw", window_size=100)
        
        models = [
            RandomForestFallDetector(random_state=42),
            CNN1DFallDetector(input_channels=6, sequence_length=100),
            LSTMFallDetector(input_size=6),
            TransformerFallDetector(input_size=6)
        ]
        
        for model in models:
            # Train model
            if isinstance(model, RandomForestFallDetector):
                model.fit(dataset.data[:100], dataset.labels[:100])
            else:
                model.fit(dataset.data[:100], dataset.labels[:100], epochs=3, batch_size=16)
            
            # Test prediction
            predictions = model.predict(dataset.data[:10])
            assert len(predictions) == 10
            assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_error_handling(self):
        """Test model error handling."""
        model = RandomForestFallDetector()
        
        # Test prediction before fitting
        with pytest.raises(ValueError):
            model.predict(np.random.randn(10, 5))
        
        with pytest.raises(ValueError):
            model.predict_proba(np.random.randn(10, 5))


if __name__ == "__main__":
    pytest.main([__file__])
