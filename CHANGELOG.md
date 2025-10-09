# Changelog

All notable changes to SAMCell will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-10-09

### Added
- **Unified Dataset Processing Framework**: New `dataset_processing/` framework with single CLI for all dataset types
  - `process_dataset.py`: Unified script supporting LIVECell, Cellpose, and custom datasets
  - `dataset_utils.py`: Shared utility functions for preprocessing, distance maps, and weight maps
  - Comprehensive documentation (README.md, QUICKSTART.md, MIGRATION_GUIDE.md)
  - Removed lab-specific code, now fully generic
- **Hugging Face Space**: Ready-to-deploy Gradio app with ZeroGPU support (`huggingface_space/`)
- **Enhanced Training Scripts**: Improved WandB logging with gradient statistics, predictions, and system metrics
- **Evaluation Framework**: Organized ablation studies and evaluation utilities in `evaluation/`

### Changed
- **Updated Citation**: Changed from bioRxiv preprint to PLOS ONE 2025 publication across all files
  - Paper DOI: https://doi.org/10.1371/journal.pone.0319532
  - Updated in README.md, CLAUDE.md, pyproject.toml
- **Improved README**: Added comprehensive method overview, training details, dataset descriptions
- **Enhanced Documentation**: Better structure with CLAUDE.md for development guidance

### Fixed
- Dataset processing now uses consistent CLAHE preprocessing (clipLimit=3.0, tileGridSize=8×8)
- Improved compatibility between dataset processing and training scripts
- Better error handling in dataset loaders

### Deprecated
- Old individual processing scripts moved to `dataset_processing/_archive/`:
  - `processLiveCell.py`
  - `processCellPose.py`
  - `process_ratcliff_lab_data.py`
  - Jupyter notebooks for dataset processing

## [1.1.4] - 2024-XX-XX

### Changed
- Initial release with core functionality
- SAMCell model implementation
- Training and evaluation scripts
- Basic dataset processing

---

**Citation:**

VandeLoo AD*, Malta NJ*, Sanganeriya S, Aponte E, van Zyl C, et al. (2025) SAMCell: Generalized label-free biological cell segmentation with segment anything. PLOS ONE 20(9): e0319532. https://doi.org/10.1371/journal.pone.0319532
