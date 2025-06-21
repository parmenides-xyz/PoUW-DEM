"""
Data Module

Contains datasets and data management utilities:
- Training and testing datasets
- HDF5 data files
- CSV data exports
- Analytics database
"""

import os
import pandas as pd

# Get the data directory path
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Common data file paths
ANALYTICS_DB = os.path.join(DATA_DIR, "pouw_dem_analytics.db")
HDF5_DATA = os.path.join(DATA_DIR, "0.h5")

# CSV data files
CSV_FILES = {
    "select_data": os.path.join(DATA_DIR, "select_data.csv"),
    "select_test_data": os.path.join(DATA_DIR, "select_test_data.csv"),
    "select_test_data_30m": os.path.join(DATA_DIR, "select_test_data_30m.csv"),
    "select_test_data_30m_2": os.path.join(DATA_DIR, "select_test_data_30m_2.csv"),
    "select_train_data": os.path.join(DATA_DIR, "select_train_data.csv"),
    "select_train_data_30m": os.path.join(DATA_DIR, "select_train_data_30m.csv"),
    "select_train_data_30m_2": os.path.join(DATA_DIR, "select_train_data_30m_2.csv"),
}

def load_csv_data(data_type):
    """Load a specific CSV data file."""
    if data_type not in CSV_FILES:
        raise ValueError(f"Unknown data type: {data_type}. Available types: {list(CSV_FILES.keys())}")
    
    file_path = CSV_FILES[data_type]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    return pd.read_csv(file_path)

def get_data_path(filename):
    """Get the full path for a data file."""
    return os.path.join(DATA_DIR, filename)

def list_available_data():
    """List all available data files."""
    files = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(('.csv', '.h5', '.db')):
            files.append({
                'name': file,
                'path': os.path.join(DATA_DIR, file),
                'size': os.path.getsize(os.path.join(DATA_DIR, file))
            })
    return files

__all__ = [
    "DATA_DIR",
    "ANALYTICS_DB",
    "HDF5_DATA",
    "CSV_FILES",
    "load_csv_data",
    "get_data_path",
    "list_available_data",
]