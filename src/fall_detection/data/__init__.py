"""Data loading and preprocessing for fall detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from .utils import (
    create_sliding_windows,
    extract_statistical_features,
    extract_frequency_features,
    normalize_features,
    set_seed,
)

logger = logging.getLogger(__name__)


class FallDetectionDataset(Dataset):
    """PyTorch Dataset for fall detection data.
    
    This dataset handles synthetic IMU sensor data for fall detection.
    It includes accelerometer and gyroscope readings with proper preprocessing.
    """
    
    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        window_size: int = 100,
        overlap: float = 0.5,
        normalize: bool = True,
        feature_extraction: str = "raw",  # "raw", "statistical", "frequency", "combined"
        transform: Optional[callable] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data: Input sensor data (samples, features).
            labels: Corresponding labels (0=normal, 1=fall).
            window_size: Size of sliding windows for time series.
            overlap: Overlap ratio between windows.
            normalize: Whether to normalize features.
            feature_extraction: Type of feature extraction to use.
            transform: Optional data transformation.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.normalize = normalize
        self.feature_extraction = feature_extraction
        self.transform = transform
        
        if data is None:
            data, labels = self._generate_synthetic_data()
        
        self.data = data
        self.labels = labels
        self.scaler = None
        
        # Preprocess data
        self._preprocess_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
        logger.info(f"Feature shape: {self.data.shape}")
        logger.info(f"Class distribution: {np.bincount(self.labels)}")
    
    def _generate_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic IMU sensor data for demonstration.
        
        Returns:
            Tuple of (data, labels) where data contains sensor readings
            and labels are binary (0=normal, 1=fall).
        """
        set_seed(42)  # For reproducible synthetic data
        
        n_samples = 1000
        n_features = 6  # 3-axis accelerometer + 3-axis gyroscope
        sequence_length = 200
        
        data = np.zeros((n_samples, sequence_length, n_features))
        labels = np.zeros(n_samples, dtype=int)
        
        # Generate normal activities (walking, standing, sitting)
        normal_samples = int(0.7 * n_samples)
        for i in range(normal_samples):
            # Simulate normal activities with small variations
            base_accel = np.array([0, 0, 9.8])  # Gravity
            base_gyro = np.array([0, 0, 0])    # No rotation
            
            # Add small random variations
            accel_noise = np.random.normal(0, 0.5, (sequence_length, 3))
            gyro_noise = np.random.normal(0, 0.1, (sequence_length, 3))
            
            data[i, :, :3] = base_accel + accel_noise
            data[i, :, 3:] = base_gyro + gyro_noise
            labels[i] = 0
        
        # Generate fall events
        fall_samples = n_samples - normal_samples
        for i in range(normal_samples, n_samples):
            # Simulate fall with sudden acceleration spike
            base_accel = np.array([0, 0, 9.8])
            base_gyro = np.array([0, 0, 0])
            
            # Add fall signature
            fall_start = np.random.randint(50, sequence_length - 50)
            fall_duration = np.random.randint(10, 30)
            
            # Sudden acceleration spike during fall
            fall_accel = np.random.normal(0, 3.0, (fall_duration, 3))
            fall_accel[:, 2] += np.random.uniform(5, 15)  # Vertical acceleration spike
            
            # Rotation during fall
            fall_gyro = np.random.normal(0, 2.0, (fall_duration, 3))
            
            # Normal activity before and after fall
            accel_noise = np.random.normal(0, 0.5, (sequence_length, 3))
            gyro_noise = np.random.normal(0, 0.1, (sequence_length, 3))
            
            data[i, :, :3] = base_accel + accel_noise
            data[i, :, 3:] = base_gyro + gyro_noise
            
            # Insert fall signature
            data[i, fall_start:fall_start+fall_duration, :3] += fall_accel
            data[i, fall_start:fall_start+fall_duration, 3:] += fall_gyro
            
            labels[i] = 1
        
        return data, labels
    
    def _preprocess_data(self) -> None:
        """Preprocess the data according to specified parameters."""
        if self.feature_extraction == "raw":
            # Use raw time series data
            if self.window_size < self.data.shape[1]:
                # Create sliding windows
                processed_data = []
                for i in range(len(self.data)):
                    windows = create_sliding_windows(
                        self.data[i], self.window_size, self.overlap
                    )
                    processed_data.extend(windows)
                
                # Repeat labels for each window
                repeated_labels = []
                for i, label in enumerate(self.labels):
                    windows_per_sample = len(create_sliding_windows(
                        self.data[i], self.window_size, self.overlap
                    ))
                    repeated_labels.extend([label] * windows_per_sample)
                
                self.data = np.array(processed_data)
                self.labels = np.array(repeated_labels)
        
        elif self.feature_extraction == "statistical":
            # Extract statistical features
            features = []
            for i in range(len(self.data)):
                feat = extract_statistical_features(self.data[i])
                features.append(feat)
            self.data = np.array(features)
        
        elif self.feature_extraction == "frequency":
            # Extract frequency features
            features = []
            for i in range(len(self.data)):
                feat = extract_frequency_features(self.data[i])
                features.append(feat)
            self.data = np.array(features)
        
        elif self.feature_extraction == "combined":
            # Combine raw, statistical, and frequency features
            processed_data = []
            for i in range(len(self.data)):
                # Raw features (flattened)
                raw_feat = self.data[i].flatten()
                
                # Statistical features
                stat_feat = extract_statistical_features(self.data[i])
                
                # Frequency features
                freq_feat = extract_frequency_features(self.data[i])
                
                # Combine all features
                combined_feat = np.concatenate([raw_feat, stat_feat, freq_feat])
                processed_data.append(combined_feat)
            
            self.data = np.array(processed_data)
        
        # Normalize features if requested
        if self.normalize:
            self.data, _ = normalize_features(self.data, method="zscore")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple of (data, label).
        """
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label
    
    def get_class_weights(self) -> np.ndarray:
        """Calculate class weights for imbalanced dataset.
        
        Returns:
            Array of class weights.
        """
        class_counts = np.bincount(self.labels)
        total_samples = len(self.labels)
        num_classes = len(class_counts)
        
        weights = total_samples / (num_classes * class_counts)
        return weights


class FallDetectionDataModule:
    """Data module for fall detection with train/validation/test splits.
    
    This class handles data loading, preprocessing, and splitting for
    fall detection experiments.
    """
    
    def __init__(
        self,
        batch_size: int = 32,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        **dataset_kwargs
    ):
        """Initialize the data module.
        
        Args:
            batch_size: Batch size for data loaders.
            test_size: Fraction of data to use for testing.
            val_size: Fraction of training data to use for validation.
            random_state: Random seed for reproducibility.
            **dataset_kwargs: Additional arguments for dataset initialization.
        """
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.dataset_kwargs = dataset_kwargs
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self) -> None:
        """Setup the data module with train/validation/test splits."""
        set_seed(self.random_state)
        
        # Create full dataset
        full_dataset = FallDetectionDataset(**self.dataset_kwargs)
        
        # Split into train and test
        train_indices, test_indices = train_test_split(
            range(len(full_dataset)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=full_dataset.labels
        )
        
        # Split train into train and validation
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=full_dataset.labels[train_indices]
        )
        
        # Create subset datasets
        self.train_dataset = self._create_subset(full_dataset, train_indices)
        self.val_dataset = self._create_subset(full_dataset, val_indices)
        self.test_dataset = self._create_subset(full_dataset, test_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"Data module setup complete:")
        logger.info(f"  Train samples: {len(self.train_dataset)}")
        logger.info(f"  Validation samples: {len(self.val_dataset)}")
        logger.info(f"  Test samples: {len(self.test_dataset)}")
    
    def _create_subset(self, full_dataset: FallDetectionDataset, indices: List[int]) -> FallDetectionDataset:
        """Create a subset of the dataset.
        
        Args:
            full_dataset: Full dataset to subset.
            indices: Indices to include in the subset.
            
        Returns:
            Subset dataset.
        """
        subset_data = full_dataset.data[indices]
        subset_labels = full_dataset.labels[indices]
        
        return FallDetectionDataset(
            data=subset_data,
            labels=subset_labels,
            window_size=full_dataset.window_size,
            overlap=full_dataset.overlap,
            normalize=False,  # Already normalized
            feature_extraction=full_dataset.feature_extraction,
            transform=full_dataset.transform,
        )
