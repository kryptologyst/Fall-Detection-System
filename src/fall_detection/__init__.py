"""Fall Detection System - Main Package.

This package provides a comprehensive fall detection system using IMU sensor data.
It includes multiple model architectures, evaluation metrics, and explainability tools.

DISCLAIMER: This is a research demonstration only, not for clinical use.
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Research Team"
__email__ = "research@example.com"

from .data import FallDetectionDataset, FallDetectionDataModule
from .models import (
    RandomForestFallDetector,
    CNN1DFallDetector,
    LSTMFallDetector,
    TransformerFallDetector,
)
from .metrics import FallDetectionMetrics
from .utils import set_seed, get_device, load_config

__all__ = [
    "FallDetectionDataset",
    "FallDetectionDataModule",
    "RandomForestFallDetector",
    "CNN1DFallDetector",
    "LSTMFallDetector",
    "TransformerFallDetector",
    "FallDetectionMetrics",
    "set_seed",
    "get_device",
    "load_config",
]
