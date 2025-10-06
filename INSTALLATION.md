# SAMCell Installation Guide

## Quick Start

### 1. Core Package (Inference Only)

For running segmentation with pre-trained models:

```bash
cd SAMCell
pip install -e .
```

This installs the minimal dependencies needed for inference.

### 2. Training

To train new models:

```bash
pip install -e .[training]
```

**Additional dependencies installed:**
- `wandb` - Experiment tracking and logging
- `psutil` - System resource monitoring
- `matplotlib` - Visualization for logging
- `h5py`, `torchvision` - Data handling

### 3. Evaluation & Ablations

To run evaluations and ablation studies:

```bash
pip install -e .[evaluation]
```

**Additional dependencies installed:**
- `stardist` - IoU-based segmentation metrics
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical visualizations

**Note:** You'll also need Cell Tracking Challenge binaries:
1. Download from: https://celltrackingchallenge.net/evaluation-methodology/
2. Place in `evaluation/cell-tracking-binaries/`
3. Make executable: `chmod +x evaluation/cell-tracking-binaries/*`

### 4. All Features

For complete functionality (training + evaluation + development):

```bash
pip install -e .[all]
```

This includes everything: training, evaluation, dev tools, GUI, and napari plugin.

## Optional Components

### GUI Application

For the standalone PyQt6 GUI:

```bash
pip install -e .[gui]
```

Then run:
```bash
cd ../SAMCell-GUI
python gui.py
```

### Napari Plugin

For the napari plugin:

```bash
pip install -e .[napari]
cd ../samcell-napari
pip install -e .
```

Then launch napari and find "SAMCell Segmentation" under Plugins.

### Development Tools

For development (pytest, black, mypy, etc.):

```bash
pip install -e .[dev]
```

## Dependency Groups Summary

| Group | Use Case | Key Packages |
|-------|----------|-------------|
| **core** | Inference only | torch, transformers, opencv, numpy |
| **training** | Model training | wandb, psutil, matplotlib |
| **evaluation** | Metrics & ablations | stardist, seaborn |
| **dev** | Development | pytest, black, mypy |
| **gui** | Standalone GUI | PyQt6 |
| **napari** | Napari plugin | napari, magicgui |
| **all** | Everything | All of the above |

## Verification

After installation, verify it works:

```bash
# Test CLI
samcell --help

# Test import
python -c "from samcell import FinetunedSAM, SlidingWindowPipeline; print('Success!')"

# Test training imports (if installed)
python -c "from samcell import log_training_metrics, init_wandb; print('Training ready!')"

# Test evaluation imports (if installed)
python -c "from samcell.evaluation.eval_utils import calculate_stardist_metrics; print('Evaluation ready!')"
```

## Troubleshooting

**ImportError: No module named 'wandb'**
- Install training dependencies: `pip install -e .[training]`

**ImportError: No module named 'stardist'**
- Install evaluation dependencies: `pip install -e .[evaluation]`

**CUDA not available**
- PyTorch should auto-detect CUDA. If not, reinstall PyTorch with CUDA support:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

**Cell Tracking Challenge metrics not working**
- Ensure binaries are downloaded and executable
- Check path in evaluation scripts matches binary location
