# setup.py
from setuptools import setup, find_packages

setup(
    name="stock-prediction-app",
    version="1.0.0",
    description="Stock Price Prediction System for Saudi Stock Market",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "pandas>=2.0.3",
        "numpy>=1.24.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "plotly>=5.15.0",
    ],
    python_requires=">=3.8",
)