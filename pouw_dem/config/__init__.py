"""
Configuration Module

Contains configuration files and utilities for the PoUW-DEM system:
- Progressus agent configuration
- PoUW system configuration
- Configuration management utilities
"""

import os
import configparser

# Get the config directory path
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration file paths
PROGRESSUS_CONFIG = os.path.join(CONFIG_DIR, "Progressus.properties")
PROGRESSUS_POUW_CONFIG = os.path.join(CONFIG_DIR, "Progressus_PoUW.properties")

def load_config(config_type="progressus"):
    """Load a configuration file."""
    config = configparser.ConfigParser()
    
    if config_type == "progressus":
        config_path = PROGRESSUS_CONFIG
    elif config_type == "progressus_pouw":
        config_path = PROGRESSUS_POUW_CONFIG
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config.read(config_path)
    return config

def get_config_value(config_type, section, key, default=None):
    """Get a specific configuration value."""
    try:
        config = load_config(config_type)
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return default

def update_config(config_type, section, key, value):
    """Update a configuration value."""
    config = load_config(config_type)
    
    if not config.has_section(section):
        config.add_section(section)
    
    config.set(section, key, str(value))
    
    # Determine which file to write to
    if config_type == "progressus":
        config_path = PROGRESSUS_CONFIG
    elif config_type == "progressus_pouw":
        config_path = PROGRESSUS_POUW_CONFIG
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    with open(config_path, 'w') as f:
        config.write(f)

__all__ = [
    "CONFIG_DIR",
    "PROGRESSUS_CONFIG",
    "PROGRESSUS_POUW_CONFIG",
    "load_config",
    "get_config_value",
    "update_config",
]