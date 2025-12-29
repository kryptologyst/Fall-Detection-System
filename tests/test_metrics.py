"""Unit tests for fall detection metrics."""

import pytest
import numpy as np
from src.fall_detection.metrics import FallDetectionMetrics


class TestFallDetectionMetrics:
    """Test cases for FallDetectionMetrics."""
    
    def test_metrics_initialization(self):
        """Test metrics calculator initialization."""
        metrics = FallDetectionMetrics()
        assert metrics.class_names == ["Normal", "Fall"]
        assert len(metrics.metrics_history) == 0
        
        # Test with custom class names
        custom_metrics = FallDetectionMetrics(class_names=["No Fall", "Fall Detected"])
        assert custom_metrics.class_names == ["No Fall", "Fall Detected"]
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        metrics = FallDetectionMetrics()
        
        # Create synthetic data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
        y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.4, 0.6],
                          [0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
        
        result = metrics.calculate_metrics(y_true, y_pred, y_proba, "Test Model")
        
        # Check that all expected metrics are present
        expected_metrics = [
            "accuracy", "precision", "recall", "f1_score",
            "sensitivity", "specificity", "ppv", "npv",
            "false_positive_rate", "false_negative_rate",
            "auroc", "auprc", "model_name"
        ]
        
        for metric in expected_metrics:
            assert metric in result
        
        # Check metric values are reasonable
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["sensitivity"] <= 1
        assert 0 <= result["specificity"] <= 1
        assert result["model_name"] == "Test Model"
        
        # Check that metrics were added to history
        assert len(metrics.metrics_history) == 1
    
    def test_calculate_metrics_without_probabilities(self):
        """Test metrics calculation without probabilities."""
        metrics = FallDetectionMetrics()
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        result = metrics.calculate_metrics(y_true, y_pred, None, "Test Model")
        
        # Check that ROC/PR metrics are set to 0 when no probabilities
        assert result["auroc"] == 0.0
        assert result["auprc"] == 0.0
        
        # Check other metrics are still calculated
        assert "accuracy" in result
        assert "sensitivity" in result
        assert "specificity" in result
    
    def test_calibration_metrics(self):
        """Test calibration metrics calculation."""
        metrics = FallDetectionMetrics()
        
        # Create perfectly calibrated probabilities
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8],
                          [0.9, 0.1], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])
        
        calib_metrics = metrics.calculate_calibration_metrics(y_true, y_proba)
        
        assert "brier_score" in calib_metrics
        assert "ece" in calib_metrics
        assert "calibration_curve" in calib_metrics
        
        # Check that metrics are reasonable
        assert calib_metrics["brier_score"] >= 0
        assert calib_metrics["ece"] >= 0
    
    def test_generate_report(self):
        """Test report generation."""
        metrics = FallDetectionMetrics()
        
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3],
                          [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]])
        
        report = metrics.generate_report(y_true, y_pred, y_proba, "Test Model")
        
        assert isinstance(report, str)
        assert "Fall Detection Evaluation Report" in report
        assert "Test Model" in report
        assert "Accuracy:" in report
        assert "Sensitivity:" in report
        assert "Confusion Matrix:" in report
    
    def test_get_leaderboard(self):
        """Test leaderboard generation."""
        metrics = FallDetectionMetrics()
        
        # Add some test metrics
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3],
                          [0.4, 0.6], [0.9, 0.1], [0.2, 0.8]])
        
        metrics.calculate_metrics(y_true, y_pred, y_proba, "Model A")
        metrics.calculate_metrics(y_true, y_pred, y_proba, "Model B")
        
        leaderboard = metrics.get_leaderboard()
        
        assert isinstance(leaderboard, str)
        assert "Fall Detection Model Leaderboard" in leaderboard
        assert "Model A" in leaderboard
        assert "Model B" in leaderboard
    
    def test_get_leaderboard_empty(self):
        """Test leaderboard with no models."""
        metrics = FallDetectionMetrics()
        leaderboard = metrics.get_leaderboard()
        
        assert leaderboard == "No models evaluated yet."
    
    def test_edge_cases(self):
        """Test edge cases in metrics calculation."""
        metrics = FallDetectionMetrics()
        
        # Test with all same predictions
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        result = metrics.calculate_metrics(y_true, y_pred, None, "Perfect Model")
        
        assert result["accuracy"] == 1.0
        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 1.0  # No negatives to test
        
        # Test with all wrong predictions
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        
        result = metrics.calculate_metrics(y_true, y_pred, None, "Wrong Model")
        
        assert result["accuracy"] == 0.0
        assert result["sensitivity"] == 0.0  # No positives to detect
        assert result["specificity"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
