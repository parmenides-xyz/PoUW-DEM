"""
Web Module

Provides web interfaces and dashboards for the PoUW-DEM system:
- Analytics dashboard for monitoring system performance
- Real-time data visualization
- Web-based control interfaces
"""

import os

# Get the web directory path
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")

# Dashboard paths
ANALYTICS_DASHBOARD = os.path.join(WEB_DIR, "analytics_dashboard.html")
MAIN_DASHBOARD = os.path.join(WEB_DIR, "dashboard.html")

def get_dashboard_path(dashboard_type="main"):
    """Get the path to a specific dashboard file."""
    if dashboard_type == "analytics":
        return ANALYTICS_DASHBOARD
    elif dashboard_type == "main":
        return MAIN_DASHBOARD
    else:
        raise ValueError(f"Unknown dashboard type: {dashboard_type}")

def get_template_path(template_name):
    """Get the path to a template file."""
    template_path = os.path.join(TEMPLATES_DIR, template_name)
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_name}")
    return template_path

__all__ = [
    "WEB_DIR",
    "TEMPLATES_DIR",
    "ANALYTICS_DASHBOARD",
    "MAIN_DASHBOARD",
    "get_dashboard_path",
    "get_template_path",
]