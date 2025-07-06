"""
Model loading utilities for production deployment
"""

import streamlit as st
import joblib
import json
import requests
import pandas as pd
from pathlib import Path
import os

# --- Configuration for your model URLs ---
MODEL_URLS = {
    "best_model.pkl": "https://github.com/ShubhamS168/Celebal-CSI-Data-Science/blob/main/Assignment/week_7/models/best_model.pkl",
    "feature_names.pkl": "https://github.com/ShubhamS168/Celebal-CSI-Data-Science/blob/main/Assignment/week_7/models/feature_names.pkl",
    "model_metadata.json": "https://github.com/ShubhamS168/Celebal-CSI-Data-Science/blob/main/Assignment/week_7/models/model_metadata.json"
}

def download_file(url, destination):
    """Downloads a file from a URL to a destination path."""
    try:
        with st.spinner(f"Downloading {destination.name}..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"✅ Downloaded {destination.name}")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to download {destination.name}: {e}")
        return False
    return True

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model, downloading it if necessary."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check for each file and download if missing
    for filename, url in MODEL_URLS.items():
        local_path = models_dir / filename
        if not local_path.exists():
            if not download_file(url, local_path):
                return None, None, None # Stop if download fails
    
    # Once all files are present, load them
    try:
        model = joblib.load(models_dir / "best_model.pkl")
        feature_names = joblib.load(models_dir / "feature_names.pkl")
        with open(models_dir / "model_metadata.json", 'r') as f:
            metadata = json.load(f)
        return model, feature_names, metadata
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


def get_model_info():
    """Get model information for display"""
    try:
        with open('models/model_metadata.json', 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception:
        return None

def check_model_availability():
    """Check if all required model files are available."""
    required_files = [Path("models") / f for f in MODEL_URLS.keys()]
    missing_files = [f.name for f in required_files if not f.exists()]
    return len(missing_files) == 0, missing_files


def get_model_version():
    """Get current model version"""
    try:
        metadata = get_model_info()
        return metadata.get('version', '1.0') if metadata else 'Unknown'
    except Exception:
        return 'Unknown'

@st.cache_data
def get_model_performance_summary():
    """Get cached model performance summary"""
    try:
        metadata = get_model_info()
        if metadata and 'all_results' in metadata:
            results_df = pd.DataFrame(metadata['all_results'])
            return {
                'best_accuracy': results_df['test_accuracy'].max(),
                'avg_accuracy': results_df['test_accuracy'].mean(),
                'model_count': len(results_df),
                'best_model': results_df.loc[results_df['test_accuracy'].idxmax(), 'model_name']
            }
    except Exception:
        pass
    return None