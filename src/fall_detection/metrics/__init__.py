"""Evaluation metrics for fall detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FallDetectionMetrics:
    """Comprehensive evaluation metrics for fall detection models.
    
    This class provides clinical and technical metrics specifically
    relevant for fall detection systems.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """Initialize the metrics calculator.
        
        Args:
            class_names: Names of classes (default: ["Normal", "Fall"]).
        """
        self.class_names = class_names or ["Normal", "Fall"]
        self.metrics_history = []
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            model_name: Name of the model for logging.
            
        Returns:
            Dictionary of calculated metrics.
        """
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="binary")
        metrics["recall"] = recall_score(y_true, y_pred, average="binary")
        metrics["f1_score"] = f1_score(y_true, y_pred, average="binary")
        
        # Clinical metrics (for fall detection, recall is critical)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Additional clinical metrics
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # ROC and PR metrics (if probabilities available)
        if y_proba is not None:
            try:
                metrics["auroc"] = roc_auc_score(y_true, y_proba[:, 1])
                metrics["auprc"] = average_precision_score(y_true, y_proba[:, 1])
            except ValueError as e:
                logger.warning(f"Could not calculate ROC/PR metrics: {e}")
                metrics["auroc"] = 0.0
                metrics["auprc"] = 0.0
        
        # Store metrics with model name
        metrics["model_name"] = model_name
        self.metrics_history.append(metrics.copy())
        
        logger.info(f"Metrics calculated for {model_name}:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        if y_proba is not None:
            logger.info(f"  AUROC: {metrics['auroc']:.4f}")
            logger.info(f"  AUPRC: {metrics['auprc']:.4f}")
        
        return metrics
    
    def calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            n_bins: Number of bins for calibration curve.
            
        Returns:
            Dictionary of calibration metrics.
        """
        # Brier Score
        brier_score = np.mean((y_proba[:, 1] - y_true) ** 2)
        
        # Expected Calibration Error (ECE)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba[:, 1], n_bins=n_bins
        )
        
        # Calculate ECE
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba[:, 1] > bin_lower) & (y_proba[:, 1] <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin, 1].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            "brier_score": brier_score,
            "ece": ece,
            "calibration_curve": (fraction_of_positives, mean_predicted_value)
        }
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            title: Plot title.
            save_path: Path to save the plot.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auroc = roc_auc_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            title: Plot title.
            save_path: Path to save the plot.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        auprc = average_precision_score(y_true, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR Curve (AUPRC = {auprc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Calibration Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            title: Plot title.
            save_path: Path to save the plot.
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba[:, 1], n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities (optional).
            model_name: Name of the model.
            
        Returns:
            Formatted evaluation report.
        """
        metrics = self.calculate_metrics(y_true, y_pred, y_proba, model_name)
        
        report = f"""
Fall Detection Evaluation Report
===============================
Model: {model_name}

Classification Metrics:
----------------------
Accuracy:     {metrics['accuracy']:.4f}
Precision:    {metrics['precision']:.4f}
Recall:       {metrics['recall']:.4f}
F1-Score:     {metrics['f1_score']:.4f}

Clinical Metrics:
----------------
Sensitivity:  {metrics['sensitivity']:.4f} (True Positive Rate)
Specificity:  {metrics['specificity']:.4f} (True Negative Rate)
PPV:          {metrics['ppv']:.4f} (Positive Predictive Value)
NPV:          {metrics['npv']:.4f} (Negative Predictive Value)
FPR:          {metrics['false_positive_rate']:.4f} (False Positive Rate)
FNR:          {metrics['false_negative_rate']:.4f} (False Negative Rate)
"""
        
        if y_proba is not None:
            report += f"""
Performance Metrics:
-------------------
AUROC:        {metrics['auroc']:.4f} (Area Under ROC Curve)
AUPRC:        {metrics['auprc']:.4f} (Area Under PR Curve)
"""
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report += f"""
Confusion Matrix:
----------------
                Predicted
Actual    {self.class_names[0]:>8} {self.class_names[1]:>8}
{self.class_names[0]:>8} {cm[0,0]:>8} {cm[0,1]:>8}
{self.class_names[1]:>8} {cm[1,0]:>8} {cm[1,1]:>8}
"""
        
        return report
    
    def get_leaderboard(self) -> str:
        """Generate a leaderboard of all evaluated models.
        
        Returns:
            Formatted leaderboard string.
        """
        if not self.metrics_history:
            return "No models evaluated yet."
        
        # Sort by AUROC (if available) or F1-score
        sorted_metrics = sorted(
            self.metrics_history,
            key=lambda x: x.get('auroc', x.get('f1_score', 0)),
            reverse=True
        )
        
        leaderboard = """
Fall Detection Model Leaderboard
===============================
Rank | Model           | Accuracy | Sensitivity | Specificity | F1-Score | AUROC  | AUPRC
-----|-----------------|----------|-------------|-------------|----------|--------|-------
"""
        
        for i, metrics in enumerate(sorted_metrics, 1):
            model_name = metrics.get('model_name', 'Unknown')
            accuracy = metrics.get('accuracy', 0)
            sensitivity = metrics.get('sensitivity', 0)
            specificity = metrics.get('specificity', 0)
            f1_score = metrics.get('f1_score', 0)
            auroc = metrics.get('auroc', 0)
            auprc = metrics.get('auprc', 0)
            
            leaderboard += f"{i:4d} | {model_name:15s} | {accuracy:8.4f} | {sensitivity:11.4f} | {specificity:11.4f} | {f1_score:8.4f} | {auroc:6.4f} | {auprc:6.4f}\n"
        
        return leaderboard
