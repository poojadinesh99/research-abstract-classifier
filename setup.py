"""
Setup script for research-abstract-classifier package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="research-abstract-classifier",
    version="1.0.0",
    author="Research Abstract Classifier Team",
    author_email="your.email@example.com",
    description="A machine learning pipeline for classifying research abstracts into academic categories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/research-abstract-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "abstract-classifier=src.inference:main",
            "abstract-train=src.train:main",
            "abstract-evaluate=src.evaluate:main",
            "abstract-api=api.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)
