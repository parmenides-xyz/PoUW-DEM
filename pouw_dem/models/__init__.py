"""
Models Module

Contains trained models and model artifacts:
- Pre-trained renewable energy aware agents
- Model checkpoints and metrics
- Model manifests for deployment
"""

import os
import json

# Model directory paths
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
RETRAINED_DIR = os.path.join(MODELS_DIR, "retrained_renewable_agents")
TRAINED_DIR = os.path.join(RETRAINED_DIR, "trained_models")

def load_model_manifest(manifest_path):
    """Load a model manifest file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)

def get_available_models():
    """Get list of available trained models."""
    models = []
    
    # Check retrained renewable agents
    if os.path.exists(RETRAINED_DIR):
        for file in os.listdir(RETRAINED_DIR):
            if file.endswith('.pt'):
                models.append({
                    'name': file.replace('.pt', ''),
                    'path': os.path.join(RETRAINED_DIR, file),
                    'type': 'renewable_aware'
                })
    
    # Check standard trained models
    if os.path.exists(TRAINED_DIR):
        for file in os.listdir(TRAINED_DIR):
            if file.endswith('.pt'):
                models.append({
                    'name': file.replace('.pt', ''),
                    'path': os.path.join(TRAINED_DIR, file),
                    'type': 'standard'
                })
    
    return models

__all__ = [
    "MODELS_DIR",
    "RETRAINED_DIR",
    "TRAINED_DIR",
    "load_model_manifest",
    "get_available_models",
]