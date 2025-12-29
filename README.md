# Fall Detection System

**RESEARCH DEMONSTRATION ONLY - NOT FOR CLINICAL USE**

## Overview

This project demonstrates AI-powered fall detection using IMU (Inertial Measurement Unit) sensor data. The system analyzes accelerometer and gyroscope readings to classify activities as "fall" or "normal" using various machine learning approaches.

## Important Disclaimer

**This is a research and educational demonstration project only.**

- **NOT FOR CLINICAL USE**: This system is not intended for diagnostic, therapeutic, or clinical decision-making purposes
- **NOT MEDICAL ADVICE**: This software does not provide medical advice, diagnosis, or treatment recommendations
- **RESEARCH DEMO ONLY**: Designed for educational and research purposes to demonstrate machine learning techniques in healthcare applications

See [DISCLAIMER.md](DISCLAIMER.md) for complete details.

## Features

- **Multiple Model Architectures**: Random Forest, 1D CNN, LSTM, Transformer-based models
- **Comprehensive Evaluation**: Clinical metrics including sensitivity, specificity, AUROC, AUPRC
- **Explainability**: SHAP values, attention maps, and uncertainty quantification
- **Interactive Demo**: Streamlit-based web interface for real-time testing
- **Synthetic Data**: Safe demonstration using simulated sensor data
- **Modern ML Stack**: PyTorch 2.x, scikit-learn, comprehensive evaluation metrics

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Fall-Detection-System.git
cd Fall-Detection-System

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.fall_detection.models import FallDetectionModel
from src.fall_detection.data import FallDetectionDataset

# Load synthetic data
dataset = FallDetectionDataset()
model = FallDetectionModel()

# Train and evaluate
model.fit(dataset.X_train, dataset.y_train)
results = model.evaluate(dataset.X_test, dataset.y_test)
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Dataset Schema

The system uses synthetic IMU sensor data with the following features:

- **Accelerometer**: x, y, z acceleration (m/s²)
- **Gyroscope**: x, y, z angular velocity (rad/s)
- **Magnitude**: Calculated motion magnitude
- **Statistical Features**: Mean, std, min, max, skewness, kurtosis
- **Frequency Features**: FFT coefficients, spectral power
- **Temporal Features**: Rolling statistics, change detection

## Model Architectures

### 1. Random Forest (Baseline)
- Traditional ML approach with engineered features
- Fast training and inference
- Good interpretability

### 2. 1D Convolutional Neural Network
- Deep learning approach for time series
- Automatic feature learning
- Robust to noise

### 3. LSTM Network
- Recurrent architecture for temporal dependencies
- Memory of long-term patterns
- Bidirectional processing

### 4. Transformer-based Model
- Attention mechanism for sequence modeling
- State-of-the-art performance
- Parallel processing capabilities

## Evaluation Metrics

### Classification Metrics
- **AUROC**: Area Under ROC Curve
- **AUPRC**: Area Under Precision-Recall Curve
- **Sensitivity**: True Positive Rate (recall)
- **Specificity**: True Negative Rate
- **PPV**: Positive Predictive Value (precision)
- **NPV**: Negative Predictive Value

### Clinical Considerations
- **Calibration**: Reliability diagrams and Brier score
- **Decision Curves**: Clinical utility analysis
- **Latency Analysis**: Detection delay measurements
- **False Alarm Rate**: Per-hour false positive rate

## Configuration

Models and training can be configured via YAML files in `configs/`:

```yaml
# configs/default.yaml
model:
  name: "1d_cnn"
  params:
    input_channels: 6
    sequence_length: 100
    num_classes: 2

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10

data:
  window_size: 100
  overlap: 0.5
  normalize: true
```

## Project Structure

```
fall-detection-system/
├── src/fall_detection/          # Main source code
│   ├── models/                  # Model implementations
│   ├── data/                    # Data loading and preprocessing
│   ├── losses/                  # Loss functions
│   ├── metrics/                 # Evaluation metrics
│   ├── utils/                   # Utility functions
│   ├── train/                   # Training scripts
│   └── eval/                    # Evaluation scripts
├── configs/                     # Configuration files
├── scripts/                     # Command-line scripts
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── assets/                      # Generated plots and results
├── demo/                        # Interactive demo
├── DISCLAIMER.md               # Important disclaimers
└── README.md                   # This file
```

## Training Commands

```bash
# Train Random Forest baseline
python scripts/train.py --config configs/random_forest.yaml

# Train 1D CNN
python scripts/train.py --config configs/1d_cnn.yaml

# Train LSTM
python scripts/train.py --config configs/lstm.yaml

# Train Transformer
python scripts/train.py --config configs/transformer.yaml

# Evaluate all models
python scripts/evaluate.py --config configs/evaluation.yaml
```

## Known Limitations

1. **Synthetic Data**: Performance metrics are based on simulated data
2. **No Clinical Validation**: Models have not been validated in clinical settings
3. **Limited Sensor Types**: Only accelerometer and gyroscope data
4. **Single Person**: No multi-person or environmental context
5. **Static Thresholds**: No adaptive thresholding for different users

## Contributing

This is a research demonstration project. Contributions should focus on:
- Educational improvements
- Code quality and documentation
- Additional evaluation metrics
- Synthetic data generation improvements

## License

MIT License - see LICENSE file for details.

## Contact

For questions about this research demonstration, please open an issue on GitHub.

---

**Remember: This is a research demonstration only. Always consult with qualified healthcare professionals for medical decisions and emergency situations.**
# Fall-Detection-System
