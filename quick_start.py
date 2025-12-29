#!/usr/bin/env python3
"""Quick start script for fall detection system.

This script provides a quick way to test the fall detection system
with different models and configurations.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from fall_detection.data import FallDetectionDataset, FallDetectionDataModule
from fall_detection.models import (
    RandomForestFallDetector,
    CNN1DFallDetector,
    LSTMFallDetector,
    TransformerFallDetector,
)
from fall_detection.metrics import FallDetectionMetrics
from fall_detection.utils import set_seed, get_device

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_demo():
    """Run a quick demonstration of the fall detection system."""
    print("=" * 60)
    print("FALL DETECTION SYSTEM - QUICK DEMO")
    print("=" * 60)
    print("DISCLAIMER: This is a research demonstration only - NOT FOR CLINICAL USE")
    print("=" * 60)
    
    # Set random seed
    set_seed(42)
    
    # Create dataset
    print("\n1. Creating synthetic dataset...")
    dataset = FallDetectionDataset(
        window_size=100,
        feature_extraction="combined",
        normalize=True
    )
    print(f"   Dataset created with {len(dataset)} samples")
    print(f"   Feature shape: {dataset.data.shape}")
    print(f"   Class distribution: {np.bincount(dataset.labels)}")
    
    # Test Random Forest
    print("\n2. Testing Random Forest model...")
    rf_model = RandomForestFallDetector(n_estimators=50, random_state=42)
    rf_model.fit(dataset.data, dataset.labels)
    
    rf_predictions = rf_model.predict(dataset.data[:100])
    rf_probabilities = rf_model.predict_proba(dataset.data[:100])
    
    metrics = FallDetectionMetrics()
    rf_metrics = metrics.calculate_metrics(
        dataset.labels[:100], rf_predictions, rf_probabilities, "Random Forest"
    )
    
    print(f"   Accuracy: {rf_metrics['accuracy']:.3f}")
    print(f"   Sensitivity: {rf_metrics['sensitivity']:.3f}")
    print(f"   Specificity: {rf_metrics['specificity']:.3f}")
    
    # Test 1D CNN
    print("\n3. Testing 1D CNN model...")
    cnn_model = CNN1DFallDetector(
        input_channels=6,
        sequence_length=100,
        hidden_dim=32,
        num_layers=2,
        device=get_device()
    )
    
    # Create raw data for CNN
    raw_dataset = FallDetectionDataset(
        window_size=100,
        feature_extraction="raw",
        normalize=True
    )
    
    cnn_model.fit(raw_dataset.data[:100], raw_dataset.labels[:100], epochs=5, batch_size=16)
    
    cnn_predictions = cnn_model.predict(raw_dataset.data[:100])
    cnn_probabilities = cnn_model.predict_proba(raw_dataset.data[:100])
    
    cnn_metrics = metrics.calculate_metrics(
        raw_dataset.labels[:100], cnn_predictions, cnn_probabilities, "1D CNN"
    )
    
    print(f"   Accuracy: {cnn_metrics['accuracy']:.3f}")
    print(f"   Sensitivity: {cnn_metrics['sensitivity']:.3f}")
    print(f"   Specificity: {cnn_metrics['specificity']:.3f}")
    
    # Generate leaderboard
    print("\n4. Model Performance Summary:")
    print("-" * 40)
    print(metrics.get_leaderboard())
    
    print("\n" + "=" * 60)
    print("QUICK DEMO COMPLETED")
    print("=" * 60)
    print("For more advanced features:")
    print("- Run 'streamlit run demo/app.py' for interactive demo")
    print("- Run 'python scripts/train.py --config configs/1d_cnn.yaml' for training")
    print("- Run 'python scripts/evaluate.py' for comprehensive evaluation")
    print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick start for fall detection system")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick demonstration"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        quick_demo()
    else:
        print("Fall Detection System - Quick Start")
        print("Use --demo to run a quick demonstration")
        print("Use --help for more options")


if __name__ == "__main__":
    main()
