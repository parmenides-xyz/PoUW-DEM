"""Setup configuration for PoUW-DEM Integration"""
from setuptools import setup, find_packages

setup(
    name="pouw-dem",
    version="0.1.0",
    description="Privacy-preserving, grid-optimizing mining pool using PoUW and FDRL",
    author="MARA Holdings",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "web3>=6.0.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "h5py>=3.8.0",
        "sqlalchemy>=2.0.0",
        "oct2py>=5.6.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pouw-dem=pouw_dem.deployment.run_full_system:main",
        ],
    },
)