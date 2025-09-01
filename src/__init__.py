"""
SAMCell: Generalized label-free biological cell segmentation with Segment Anything

This package provides tools for automated cell segmentation in microscopy images
using a fine-tuned version of Meta's Segment Anything Model (SAM).

Key Features:
- State-of-the-art cell segmentation performance
- Zero-shot generalization to new cell types and imaging conditions
- Comprehensive morphological and intensity metrics
- Easy-to-use Python API
- Compatible with various image formats

Example usage:
    >>> import samcell
    >>> from samcell import SAMCellPipeline, FinetunedSAM
    >>> 
    >>> # Load model
    >>> model = FinetunedSAM('facebook/sam-vit-base')
    >>> model.load_weights('path/to/samcell_weights.pt')
    >>> 
    >>> # Create pipeline
    >>> pipeline = SAMCellPipeline(model, 'cuda')
    >>> 
    >>> # Segment cells
    >>> labels = pipeline.run(image)
    >>> 
    >>> # Calculate metrics
    >>> metrics = pipeline.calculate_metrics(labels, image)
"""

__version__ = "1.0.0"
__author__ = "SAMCell Team"
__email__ = "saahilsanganeriya@gatech.edu"

# Import main classes and functions
from .model import FinetunedSAM
from .pipeline import SlidingWindowPipeline
from .metrics import (
    calculate_all_metrics,
    calculate_basic_metrics, 
    calculate_neighbor_metrics,
    calculate_summary_metrics,
    export_metrics_csv,
    compute_gui_metrics,
)
from .utils import lr_warmup, init_wandb, log_wandb

# Convenience alias
SAMCellPipeline = SlidingWindowPipeline

__all__ = [
    "FinetunedSAM",
    "SlidingWindowPipeline", 
    "SAMCellPipeline",
    "calculate_all_metrics",
    "calculate_basic_metrics",
    "calculate_neighbor_metrics", 
    "calculate_summary_metrics",
    "export_metrics_csv",
    "compute_gui_metrics",
    "lr_warmup",
    "init_wandb",
    "log_wandb",
]
