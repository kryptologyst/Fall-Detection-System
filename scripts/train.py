#!/usr/bin/env python3
"""Training script for fall detection models.

This script trains various fall detection models using the specified configuration.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.fall_detection.data import FallDetectionDataModule
from src.fall_detection.models import (
    RandomForestFallDetector,
    CNN1DFallDetector,
    LSTMFallDetector,
    TransformerFallDetector,
)
from src.fall_detection.metrics import FallDetectionMetrics
from src.fall_detection.utils import set_seed, get_device, load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_model(config: DictConfig) -> Any:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        Initialized model.
    """
    model_name = config.model.name
    model_params = config.model.params[model_name]
    
    if model_name == "random_forest":
        return RandomForestFallDetector(**model_params)
    elif model_name == "1d_cnn":
        return CNN1DFallDetector(**model_params)
    elif model_name == "lstm":
        return LSTMFallDetector(**model_params)
    elif model_name == "transformer":
        return TransformerFallDetector(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_model(model: Any, data_module: FallDetectionDataModule, config: DictConfig) -> Dict[str, Any]:
    """Train the model.
    
    Args:
        model: Model to train.
        data_module: Data module with train/val/test splits.
        config: Training configuration.
        
    Returns:
        Training results.
    """
    logger.info(f"Training {config.model.name} model")
    
    # Get training data
    train_data = data_module.train_dataset.data
    train_labels = data_module.train_dataset.labels
    
    # Train the model
    if config.model.name == "random_forest":
        model.fit(train_data, train_labels)
    else:
        # Deep learning models
        epochs = config.training.epochs
        batch_size = config.data.batch_size
        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)
    
    logger.info("Training completed")
    
    return {"model": model, "config": config}


def evaluate_model(model: Any, data_module: FallDetectionDataModule, config: DictConfig) -> Dict[str, Any]:
    """Evaluate the trained model.
    
    Args:
        model: Trained model.
        data_module: Data module with test data.
        config: Configuration.
        
    Returns:
        Evaluation results.
    """
    logger.info("Evaluating model")
    
    # Get test data
    test_data = data_module.test_dataset.data
    test_labels = data_module.test_dataset.labels
    
    # Make predictions
    predictions = model.predict(test_data)
    probabilities = model.predict_proba(test_data) if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics_calculator = FallDetectionMetrics()
    metrics = metrics_calculator.calculate_metrics(
        test_labels, predictions, probabilities, config.model.name
    )
    
    # Generate report
    report = metrics_calculator.generate_report(
        test_labels, predictions, probabilities, config.model.name
    )
    
    logger.info(f"Evaluation completed for {config.model.name}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"Test Specificity: {metrics['specificity']:.4f}")
    
    return {
        "metrics": metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "report": report,
        "metrics_calculator": metrics_calculator
    }


def save_results(results: Dict[str, Any], config: DictConfig) -> None:
    """Save training and evaluation results.
    
    Args:
        results: Results dictionary.
        config: Configuration.
    """
    # Create output directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.evaluation.plots_dir, exist_ok=True)
    
    model_name = config.model.name
    
    # Save model (if it's a PyTorch model)
    if hasattr(results["model"], "model"):
        model_path = os.path.join(config.training.checkpoint_dir, f"{model_name}_model.pth")
        torch.save(results["model"].model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(config.training.checkpoint_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(results["report"])
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Generate and save plots
    if results["probabilities"] is not None:
        metrics_calculator = results["metrics_calculator"]
        test_labels = results["metrics_calculator"].metrics_history[-1].get("y_true", None)
        
        if test_labels is not None:
            # Plot confusion matrix
            cm_path = os.path.join(config.evaluation.plots_dir, f"{model_name}_confusion_matrix.png")
            metrics_calculator.plot_confusion_matrix(
                test_labels, results["predictions"], 
                f"{model_name} Confusion Matrix", cm_path
            )
            
            # Plot ROC curve
            roc_path = os.path.join(config.evaluation.plots_dir, f"{model_name}_roc_curve.png")
            metrics_calculator.plot_roc_curve(
                test_labels, results["probabilities"],
                f"{model_name} ROC Curve", roc_path
            )
            
            # Plot PR curve
            pr_path = os.path.join(config.evaluation.plots_dir, f"{model_name}_pr_curve.png")
            metrics_calculator.plot_precision_recall_curve(
                test_labels, results["probabilities"],
                f"{model_name} Precision-Recall Curve", pr_path
            )
            
            # Plot calibration curve
            cal_path = os.path.join(config.evaluation.plots_dir, f"{model_name}_calibration_curve.png")
            metrics_calculator.plot_calibration_curve(
                test_labels, results["probabilities"],
                f"{model_name} Calibration Curve", cal_path
            )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train fall detection models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Update config paths
    config.training.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    config.evaluation.plots_dir = os.path.join(args.output_dir, "plots")
    config.logging.log_dir = os.path.join(args.output_dir, "logs")
    
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Setup data
        data_module = FallDetectionDataModule(**config.data)
        data_module.setup()
        
        # Create model
        model = create_model(config)
        
        # Train model
        train_results = train_model(model, data_module, config)
        
        # Evaluate model
        eval_results = evaluate_model(model, data_module, config)
        
        # Combine results
        results = {**train_results, **eval_results}
        
        # Save results
        save_results(results, config)
        
        # Print final report
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(results["report"])
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
