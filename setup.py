"""
Setup script for SAMCell package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="samcell",
    version="1.1.2",
    author="SAMCell Team",
    author_email="saahilsanganeriya@gatech.edu",
    description="Generalized label-free biological cell segmentation with Segment Anything",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saahilsanganeriya/SAMCell",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "training": [
            "h5py>=3.0.0",
            "torchvision>=0.10.0", 
            "wandb>=0.12.0",
        ],
        "excel": [
            "openpyxl>=3.0.0",
        ],
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "jupyter>=1.0.0",
        ],
        "gui": [
            "PyQt6>=6.0",
        ],
        "napari": [
            "napari>=0.4.14",
            "magicgui>=0.5.0",
        ],
        "all": [
            "samcell[training,excel,dev,gui,napari]",
        ],
    },
    entry_points={
        "console_scripts": [
            "samcell=samcell.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
