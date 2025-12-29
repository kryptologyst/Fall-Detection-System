"""Utility functions for fall detection system."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        OmegaConf configuration object.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        config_path: Path to save configuration file.
    """
    OmegaConf.save(config, config_path)


def normalize_features(
    data: np.ndarray, 
    method: str = "zscore",
    stats: Optional[Dict[str, float]] = None
) -> tuple[np.ndarray, Dict[str, float]]:
    """Normalize features using specified method.
    
    Args:
        data: Input data array.
        method: Normalization method ('zscore', 'minmax', 'robust').
        stats: Pre-computed statistics for normalization.
        
    Returns:
        Tuple of normalized data and statistics used.
    """
    if stats is None:
        if method == "zscore":
            stats = {
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0)
            }
            normalized = (data - stats["mean"]) / (stats["std"] + 1e-8)
        elif method == "minmax":
            stats = {
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0)
            }
            normalized = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
        elif method == "robust":
            stats = {
                "median": np.median(data, axis=0),
                "mad": np.median(np.abs(data - np.median(data, axis=0)), axis=0)
            }
            normalized = (data - stats["median"]) / (stats["mad"] + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    else:
        if method == "zscore":
            normalized = (data - stats["mean"]) / (stats["std"] + 1e-8)
        elif method == "minmax":
            normalized = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
        elif method == "robust":
            normalized = (data - stats["median"]) / (stats["mad"] + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, stats


def create_sliding_windows(
    data: np.ndarray, 
    window_size: int, 
    overlap: float = 0.5
) -> np.ndarray:
    """Create sliding windows from time series data.
    
    Args:
        data: Input time series data.
        window_size: Size of each window.
        overlap: Overlap ratio between windows.
        
    Returns:
        Array of sliding windows.
    """
    step_size = int(window_size * (1 - overlap))
    windows = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)


def extract_statistical_features(data: np.ndarray) -> np.ndarray:
    """Extract statistical features from time series data.
    
    Args:
        data: Input time series data.
        
    Returns:
        Array of statistical features.
    """
    features = []
    
    # Basic statistics
    features.extend([
        np.mean(data, axis=0),
        np.std(data, axis=0),
        np.min(data, axis=0),
        np.max(data, axis=0),
        np.median(data, axis=0)
    ])
    
    # Higher-order moments
    features.extend([
        np.mean((data - np.mean(data, axis=0))**2, axis=0),  # variance
        np.mean((data - np.mean(data, axis=0))**3, axis=0),  # skewness
        np.mean((data - np.mean(data, axis=0))**4, axis=0),  # kurtosis
    ])
    
    # Range and interquartile range
    features.extend([
        np.max(data, axis=0) - np.min(data, axis=0),  # range
        np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0),  # IQR
    ])
    
    return np.concatenate(features, axis=0)


def extract_frequency_features(data: np.ndarray, sampling_rate: float = 50.0) -> np.ndarray:
    """Extract frequency domain features from time series data.
    
    Args:
        data: Input time series data.
        sampling_rate: Sampling rate in Hz.
        
    Returns:
        Array of frequency features.
    """
    features = []
    
    for channel in range(data.shape[1]):
        # FFT
        fft = np.fft.fft(data[:, channel])
        fft_magnitude = np.abs(fft[:len(fft)//2])
        
        # Spectral features
        features.extend([
            np.sum(fft_magnitude),  # total power
            np.max(fft_magnitude),  # peak power
            np.mean(fft_magnitude),  # mean power
            np.std(fft_magnitude),  # power std
        ])
        
        # Dominant frequency
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)[:len(fft_magnitude)]
        dominant_freq_idx = np.argmax(fft_magnitude)
        features.append(freqs[dominant_freq_idx])
    
    return np.array(features)
