from setuptools import setup, find_packages

setup(
    name="tlcc-anomaly-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="TLCC-Based Anomaly Detection for Time Series Data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scipy>=1.6.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
