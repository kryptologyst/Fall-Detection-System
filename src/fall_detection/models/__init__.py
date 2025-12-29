"""Model implementations for fall detection."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from .utils import get_device, set_seed

logger = logging.getLogger(__name__)


class BaseFallDetector(ABC):
    """Abstract base class for fall detection models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass


class RandomForestFallDetector(BaseFallDetector):
    """Random Forest classifier for fall detection.
    
    This is a traditional machine learning approach that works well
    with engineered features and provides good interpretability.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
    ):
        """Initialize the Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            random_state: Random seed for reproducibility.
            class_weight: Class weight strategy.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.class_weight = class_weight
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight=class_weight,
        )
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model.
        
        Args:
            X: Training features.
            y: Training labels.
        """
        logger.info(f"Training Random Forest with {self.n_estimators} estimators")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Random Forest training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores.
        
        Returns:
            Feature importance array.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.model.feature_importances_


class CNN1DFallDetector(BaseFallDetector):
    """1D Convolutional Neural Network for fall detection.
    
    This model uses 1D convolutions to process time series sensor data
    and automatically learn relevant features for fall detection.
    """
    
    def __init__(
        self,
        input_channels: int = 6,
        sequence_length: int = 100,
        num_classes: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """Initialize the 1D CNN model.
        
        Args:
            input_channels: Number of input channels (sensor axes).
            sequence_length: Length of input sequences.
            num_classes: Number of output classes.
            hidden_dim: Hidden dimension size.
            num_layers: Number of convolutional layers.
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            device: Device to use for training.
        """
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device or get_device()
        
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """Build the 1D CNN model architecture."""
        layers = []
        
        # Input layer
        layers.append(nn.Conv1d(self.input_channels, self.hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout))
        
        # Hidden layers
        for i in range(self.num_layers - 1):
            layers.append(nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        
        # Output layer
        layers.append(nn.Linear(self.hidden_dim, self.num_classes))
        
        return nn.Sequential(*layers).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        """Train the 1D CNN model.
        
        Args:
            X: Training features.
            y: Training labels.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        logger.info(f"Training 1D CNN for {epochs} epochs")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Reshape for 1D CNN (batch, channels, sequence)
        if len(X_tensor.shape) == 2:
            # If features are flattened, reshape to (batch, channels, sequence)
            X_tensor = X_tensor.view(-1, self.input_channels, self.sequence_length)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("1D CNN training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.view(-1, self.input_channels, self.sequence_length)
            
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            if len(X_tensor.shape) == 2:
                X_tensor = X_tensor.view(-1, self.input_channels, self.sequence_length)
            
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()


class LSTMFallDetector(BaseFallDetector):
    """LSTM-based model for fall detection.
    
    This model uses LSTM layers to capture temporal dependencies
    in sensor data for improved fall detection.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """Initialize the LSTM model.
        
        Args:
            input_size: Number of input features per timestep.
            hidden_size: Hidden state size.
            num_layers: Number of LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            bidirectional: Whether to use bidirectional LSTM.
            learning_rate: Learning rate for optimizer.
            device: Device to use for training.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate = learning_rate
        self.device = device or get_device()
        
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM model architecture."""
        model = nn.Sequential(
            nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            ),
            nn.Dropout(self.dropout),
            nn.Linear(
                self.hidden_size * (2 if self.bidirectional else 1),
                self.num_classes
            )
        ).to(self.device)
        
        return model
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        """Train the LSTM model.
        
        Args:
            X: Training features.
            y: Training labels.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        logger.info(f"Training LSTM for {epochs} epochs")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass through LSTM
                lstm_out, _ = self.model[0](batch_X)
                # Take the last output
                lstm_out = lstm_out[:, -1, :]
                # Apply dropout and final layer
                output = self.model[1](self.model[2](lstm_out))
                
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("LSTM training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            lstm_out, _ = self.model[0](X_tensor)
            lstm_out = lstm_out[:, -1, :]
            output = self.model[1](self.model[2](lstm_out))
            predictions = torch.argmax(output, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            lstm_out, _ = self.model[0](X_tensor)
            lstm_out = lstm_out[:, -1, :]
            output = self.model[1](self.model[2](lstm_out))
            probabilities = F.softmax(output, dim=1)
            return probabilities.cpu().numpy()


class TransformerFallDetector(BaseFallDetector):
    """Transformer-based model for fall detection.
    
    This model uses a simplified transformer architecture to process
    time series sensor data with attention mechanisms.
    """
    
    def __init__(
        self,
        input_size: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        device: Optional[str] = None,
    ):
        """Initialize the Transformer model.
        
        Args:
            input_size: Number of input features per timestep.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            num_classes: Number of output classes.
            dropout: Dropout rate.
            learning_rate: Learning rate for optimizer.
            device: Device to use for training.
        """
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device or get_device()
        
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        self.is_fitted = False
    
    def _build_model(self) -> nn.Module:
        """Build the Transformer model architecture."""
        class TransformerFallDetector(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout):
                super().__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.classifier = nn.Linear(d_model, num_classes)
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                # Input projection
                x = self.input_projection(x)
                
                # Add positional encoding
                seq_len = x.size(1)
                x = x + self.pos_encoding[:seq_len].unsqueeze(0)
                
                # Transformer encoding
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Classification
                x = self.dropout(x)
                x = self.classifier(x)
                
                return x
        
        return TransformerFallDetector(
            self.input_size, self.d_model, self.nhead, 
            self.num_layers, self.num_classes, self.dropout
        ).to(self.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
        """Train the Transformer model.
        
        Args:
            X: Training features.
            y: Training labels.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        logger.info(f"Training Transformer for {epochs} epochs")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("Transformer training completed")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
            return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Input features.
            
        Returns:
            Class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
