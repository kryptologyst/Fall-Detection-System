"""Streamlit demo for fall detection system.

This interactive demo allows users to test the fall detection system
with synthetic sensor data and visualize results.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Tuple

# Set page config
st.set_page_config(
    page_title="Fall Detection System - Research Demo",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fall_detection.data import FallDetectionDataset
from fall_detection.models import (
    RandomForestFallDetector,
    CNN1DFallDetector,
    LSTMFallDetector,
    TransformerFallDetector,
)
from fall_detection.metrics import FallDetectionMetrics
from fall_detection.utils import set_seed, get_device

# Set random seed for reproducibility
set_seed(42)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">⚠️ Fall Detection System</h1>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h3>⚠️ IMPORTANT DISCLAIMER</h3>
    <p><strong>This is a research demonstration only - NOT FOR CLINICAL USE</strong></p>
    <ul>
        <li>This system is not intended for diagnostic, therapeutic, or clinical decision-making</li>
        <li>This software does not provide medical advice, diagnosis, or treatment recommendations</li>
        <li>Performance metrics are based on synthetic data and may not reflect real-world performance</li>
        <li>Always consult with qualified healthcare professionals for medical decisions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Random Forest", "1D CNN", "LSTM", "Transformer"],
    help="Choose the machine learning model for fall detection"
)

# Data parameters
st.sidebar.subheader("Data Parameters")
window_size = st.sidebar.slider("Window Size", 50, 200, 100, help="Size of time windows for analysis")
feature_type = st.sidebar.selectbox(
    "Feature Type",
    ["raw", "statistical", "frequency", "combined"],
    help="Type of features to extract from sensor data"
)

# Model parameters
st.sidebar.subheader("Model Parameters")
if model_type == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 50, 300, 100)
    max_depth = st.sidebar.slider("Max Depth", 5, 20, 10)
elif model_type == "1D CNN":
    hidden_dim = st.sidebar.slider("Hidden Dimension", 32, 256, 64)
    num_layers = st.sidebar.slider("Number of Layers", 2, 6, 3)
elif model_type == "LSTM":
    hidden_size = st.sidebar.slider("Hidden Size", 32, 128, 64)
    num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
elif model_type == "Transformer":
    d_model = st.sidebar.slider("Model Dimension", 32, 128, 64)
    nhead = st.sidebar.slider("Number of Heads", 2, 8, 4)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Results", "🔍 Explainability"])

with tab1:
    st.header("Synthetic Sensor Data Overview")
    
    # Generate and display sample data
    dataset = FallDetectionDataset(
        window_size=window_size,
        feature_extraction=feature_type,
        normalize=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Statistics")
        st.metric("Total Samples", len(dataset))
        st.metric("Normal Activities", np.sum(dataset.labels == 0))
        st.metric("Fall Events", np.sum(dataset.labels == 1))
        st.metric("Fall Rate", f"{np.mean(dataset.labels):.1%}")
    
    with col2:
        st.subheader("Feature Information")
        st.metric("Feature Shape", str(dataset.data.shape))
        st.metric("Data Type", dataset.feature_extraction)
        st.metric("Window Size", dataset.window_size)
        st.metric("Normalized", "Yes" if dataset.normalize else "No")
    
    # Visualize sample sensor data
    st.subheader("Sample Sensor Data")
    
    # Select a sample to visualize
    sample_idx = st.selectbox("Select Sample", range(min(10, len(dataset))))
    
    sample_data, sample_label = dataset[sample_idx]
    activity_type = "Fall" if sample_label == 1 else "Normal"
    
    st.write(f"**Sample {sample_idx}**: {activity_type} Activity")
    
    if len(sample_data.shape) == 1:
        # Statistical features - show as bar chart
        feature_names = [f"Feature {i}" for i in range(len(sample_data))]
        df_features = pd.DataFrame({
            "Feature": feature_names,
            "Value": sample_data
        })
        
        fig = px.bar(df_features, x="Feature", y="Value", title="Statistical Features")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Time series data - show as line plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Accelerometer (m/s²)", "Gyroscope (rad/s)"],
            vertical_spacing=0.1
        )
        
        # Accelerometer data
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    y=sample_data[:, i],
                    mode='lines',
                    name=f'Accel {["X", "Y", "Z"][i]}',
                    line=dict(color=['red', 'green', 'blue'][i])
                ),
                row=1, col=1
            )
        
        # Gyroscope data
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    y=sample_data[:, i+3],
                    mode='lines',
                    name=f'Gyro {["X", "Y", "Z"][i]}',
                    line=dict(color=['red', 'green', 'blue'][i])
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=600, title_text=f"Sensor Data - {activity_type} Activity")
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Model Training")
    
    # Create model based on selection
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Create model
            if model_type == "Random Forest":
                model = RandomForestFallDetector(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif model_type == "1D CNN":
                model = CNN1DFallDetector(
                    input_channels=6,
                    sequence_length=window_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    device=get_device()
                )
            elif model_type == "LSTM":
                model = LSTMFallDetector(
                    input_size=6,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    device=get_device()
                )
            elif model_type == "Transformer":
                model = TransformerFallDetector(
                    input_size=6,
                    d_model=d_model,
                    nhead=nhead,
                    device=get_device()
                )
            
            # Train model
            start_time = time.time()
            
            if model_type == "Random Forest":
                model.fit(dataset.data, dataset.labels)
            else:
                model.fit(dataset.data, dataset.labels, epochs=50, batch_size=32)
            
            training_time = time.time() - start_time
            
            # Store model in session state
            st.session_state.model = model
            st.session_state.model_type = model_type
            st.session_state.training_time = training_time
            
            st.success(f"✅ {model_type} model trained successfully!")
            st.info(f"Training time: {training_time:.2f} seconds")

with tab3:
    st.header("Model Results")
    
    if "model" in st.session_state:
        model = st.session_state.model
        model_type = st.session_state.model_type
        
        # Make predictions
        predictions = model.predict(dataset.data)
        probabilities = model.predict_proba(dataset.data) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics_calc = FallDetectionMetrics()
        metrics = metrics_calc.calculate_metrics(
            dataset.labels, predictions, probabilities, model_type
        )
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("Precision", f"{metrics['precision']:.3f}")
        
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
            st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
        
        with col3:
            st.metric("Sensitivity", f"{metrics['sensitivity']:.3f}")
            st.metric("Specificity", f"{metrics['specificity']:.3f}")
        
        with col4:
            if probabilities is not None:
                st.metric("AUROC", f"{metrics['auroc']:.3f}")
                st.metric("AUPRC", f"{metrics['auprc']:.3f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = metrics_calc.calculate_metrics(dataset.labels, predictions, probabilities, model_type)
        
        # Create confusion matrix plot
        confusion_data = np.array([[metrics['specificity'] * (1-metrics['false_positive_rate']), 
                                   metrics['false_positive_rate']],
                                  [metrics['false_negative_rate'], 
                                   metrics['sensitivity']]])
        
        fig = px.imshow(confusion_data, 
                       text_auto=True,
                       aspect="auto",
                       labels=dict(x="Predicted", y="Actual"),
                       x=["Normal", "Fall"],
                       y=["Normal", "Fall"],
                       title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # ROC Curve (if probabilities available)
        if probabilities is not None:
            st.subheader("ROC Curve")
            
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(dataset.labels, probabilities[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                   name=f'ROC Curve (AUC = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                                   name='Random Classifier', line=dict(dash='dash')))
            fig.update_layout(title="ROC Curve",
                             xaxis_title="False Positive Rate",
                             yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        st.subheader("Model Performance Summary")
        
        performance_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Sensitivity", "Specificity"],
            "Value": [metrics['accuracy'], metrics['precision'], metrics['recall'], 
                      metrics['f1_score'], metrics['sensitivity'], metrics['specificity']]
        }
        
        df_performance = pd.DataFrame(performance_data)
        fig = px.bar(df_performance, x="Metric", y="Value", title="Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("Please train a model first using the 'Model Training' tab.")

with tab4:
    st.header("Model Explainability")
    
    if "model" in st.session_state:
        model = st.session_state.model
        model_type = st.session_state.model_type
        
        st.subheader("Feature Importance")
        
        if model_type == "Random Forest":
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                
                # Create feature importance plot
                feature_names = [f"Feature {i}" for i in range(len(importance))]
                df_importance = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                })
                
                fig = px.bar(df_importance, x="Feature", y="Importance", 
                           title="Feature Importance (Random Forest)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Sample prediction explanation
        st.subheader("Sample Prediction Analysis")
        
        sample_idx = st.selectbox("Select Sample for Analysis", range(min(20, len(dataset))))
        sample_data, sample_label = dataset[sample_idx]
        
        # Make prediction
        pred = model.predict(sample_data.reshape(1, -1))[0]
        proba = model.predict_proba(sample_data.reshape(1, -1))[0] if hasattr(model, 'predict_proba') else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**True Label**: {'Fall' if sample_label == 1 else 'Normal'}")
            st.write(f"**Predicted**: {'Fall' if pred == 1 else 'Normal'}")
            st.write(f"**Correct**: {'✅' if pred == sample_label else '❌'}")
        
        with col2:
            if proba is not None:
                st.write(f"**Confidence**: {max(proba):.3f}")
                st.write(f"**Normal Prob**: {proba[0]:.3f}")
                st.write(f"**Fall Prob**: {proba[1]:.3f}")
        
        # Show sample data
        if len(sample_data.shape) > 1:
            st.subheader("Sensor Data Visualization")
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Accelerometer", "Gyroscope"],
                vertical_spacing=0.1
            )
            
            # Accelerometer
            for i in range(3):
                fig.add_trace(
                    go.Scatter(
                        y=sample_data[:, i],
                        mode='lines',
                        name=f'Accel {["X", "Y", "Z"][i]}',
                        line=dict(color=['red', 'green', 'blue'][i])
                    ),
                    row=1, col=1
                )
            
            # Gyroscope
            for i in range(3):
                fig.add_trace(
                    go.Scatter(
                        y=sample_data[:, i+3],
                        mode='lines',
                        name=f'Gyro {["X", "Y", "Z"][i]}',
                        line=dict(color=['red', 'green', 'blue'][i])
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(height=500, title_text="Sample Sensor Data")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please train a model first using the 'Model Training' tab.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>Fall Detection System - Research Demonstration</strong></p>
    <p>⚠️ This is a research demonstration only - NOT FOR CLINICAL USE</p>
    <p>Always consult with qualified healthcare professionals for medical decisions</p>
</div>
""", unsafe_allow_html=True)
