from setuptools import setup, find_packages

setup(
    name="asclepius-biosignal",
    version="1.0.0",
    description=(
        "ASCLEPIUS — Adaptive Signal Classification and Learning Engine "
        "for Predictive Intelligent Universal Signal analysis"
    ),
    author="ASCLEPIUS Contributors",
    license="MIT",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "mne>=1.5.0",
        "neurokit2>=0.2.7",
        "lightgbm>=4.1.0",
        "shap>=0.43.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "rich>=13.6.0",
        "loguru>=0.7.2",
    ],
    entry_points={
        "console_scripts": [
            "asclepius=asclepius.cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
