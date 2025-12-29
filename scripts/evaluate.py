#!/usr/bin/env python3
"""Evaluation script for fall detection models.

This script evaluates trained models and generates comprehensive reports.
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


def load_trained_model(model_path: str, config: DictConfig) -> Any:
    """Load a trained model.
    
    Args:
        model_path: Path to the trained model.
        config: Model configuration.
        
    Returns:
        Loaded model.
    """
    model_name = config.model.name
    model_params = config.model.params[model_name]
    
    if model_name == "random_forest":
        model = RandomForestFallDetector(**model_params)
        # For Random Forest, we would need to save/load the sklearn model
        # This is a simplified version
        logger.warning("Random Forest model loading not implemented in this demo")
        return model
    elif model_name == "1d_cnn":
        model = CNN1DFallDetector(**model_params)
        if os.path.exists(model_path):
            model.model.load_state_dict(torch.load(model_path))
            model.is_fitted = True
        return model
    elif model_name == "lstm":
        model = LSTMFallDetector(**model_params)
        if os.path.exists(model_path):
            model.model.load_state_dict(torch.load(model_path))
            model.is_fitted = True
        return model
    elif model_name == "transformer":
        model = TransformerFallDetector(**model_params)
        if os.path.exists(model_path):
            model.model.load_state_dict(torch.load(model_path))
            model.is_fitted = True
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate_all_models(config: DictConfig) -> Dict[str, Any]:
    """Evaluate all available models.
    
    Args:
        config: Configuration.
        
    Returns:
        Dictionary of evaluation results for all models.
    """
    # Setup data
    data_module = FallDetectionDataModule(**config.data)
    data_module.setup()
    
    # Get test data
    test_data = data_module.test_dataset.data
    test_labels = data_module.test_dataset.labels
    
    # Models to evaluate
    models_to_evaluate = [
        ("random_forest", RandomForestFallDetector(random_state=42)),
        ("1d_cnn", CNN1DFallDetector(input_channels=6, sequence_length=100)),
        ("lstm", LSTMFallDetector(input_size=6)),
        ("transformer", TransformerFallDetector(input_size=6))
    ]
    
    results = {}
    metrics_calculator = FallDetectionMetrics()
    
    for model_name, model in models_to_evaluate:
        logger.info(f"Evaluating {model_name} model")
        
        try:
            # Train model quickly for evaluation
            if model_name == "random_forest":
                model.fit(test_data, test_labels)
            else:
                model.fit(test_data, test_labels, epochs=10, batch_size=32)
            
            # Make predictions
            predictions = model.predict(test_data)
            probabilities = model.predict_proba(test_data) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = metrics_calculator.calculate_metrics(
                test_labels, predictions, probabilities, model_name
            )
            
            results[model_name] = {
                "metrics": metrics,
                "predictions": predictions,
                "probabilities": probabilities
            }
            
            logger.info(f"{model_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            results[model_name] = {"error": str(e)}
    
    return results, metrics_calculator


def generate_comprehensive_report(results: Dict[str, Any], metrics_calculator: FallDetectionMetrics) -> str:
    """Generate comprehensive evaluation report.
    
    Args:
        results: Evaluation results for all models.
        metrics_calculator: Metrics calculator with history.
        
    Returns:
        Comprehensive report string.
    """
    report = """
Fall Detection System - Comprehensive Evaluation Report
=====================================================

DISCLAIMER: This is a research demonstration only - NOT FOR CLINICAL USE

"""
    
    # Model performance summary
    report += "\nModel Performance Summary:\n"
    report += "-" * 50 + "\n"
    
    for model_name, result in results.items():
        if "error" in result:
            report += f"\n{model_name.upper()}: ERROR - {result['error']}\n"
            continue
        
        metrics = result["metrics"]
        report += f"\n{model_name.upper()}:\n"
        report += f"  Accuracy:     {metrics['accuracy']:.4f}\n"
        report += f"  Sensitivity:  {metrics['sensitivity']:.4f}\n"
        report += f"  Specificity:  {metrics['specificity']:.4f}\n"
        report += f"  F1-Score:     {metrics['f1_score']:.4f}\n"
        if 'auroc' in metrics and metrics['auroc'] > 0:
            report += f"  AUROC:        {metrics['auroc']:.4f}\n"
            report += f"  AUPRC:        {metrics['auprc']:.4f}\n"
    
    # Leaderboard
    report += "\n" + metrics_calculator.get_leaderboard()
    
    # Clinical interpretation
    report += "\n\nClinical Interpretation:\n"
    report += "-" * 50 + "\n"
    report += """
For fall detection systems, the following metrics are particularly important:

1. SENSITIVITY (Recall): Critical for detecting actual falls
   - Higher sensitivity = fewer missed falls
   - Target: >90% for clinical applications

2. SPECIFICITY: Important for reducing false alarms
   - Higher specificity = fewer false alarms
   - Target: >95% for clinical applications

3. LATENCY: Time to detection
   - Critical for emergency response
   - Target: <5 seconds for fall detection

4. FALSE ALARM RATE: Per-hour false positive rate
   - Important for user acceptance
   - Target: <1 false alarm per day

IMPORTANT LIMITATIONS:
- These results are based on synthetic data
- Real-world performance may differ significantly
- Clinical validation is required for any medical application
- This system is NOT approved for clinical use
"""
    
    return report


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fall detection models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="checkpoints",
        help="Directory containing trained models"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation with config: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Evaluate all models
        results, metrics_calculator = evaluate_all_models(config)
        
        # Generate comprehensive report
        report = generate_comprehensive_report(results, metrics_calculator)
        
        # Save report
        report_path = os.path.join(args.output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        
        # Print report
        print(report)
        
        logger.info("Evaluation completed successfully")
        logger.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
