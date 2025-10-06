# SAMCell: Generalized Label-Free Biological Cell Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-PLOS%20ONE-red.svg)](https://doi.org/10.1371/journal.pone.0319532)

SAMCell is a state-of-the-art deep learning model for automated cell segmentation in microscopy images. Built on Meta's Segment Anything Model (SAM), SAMCell uses a novel distance map regression approach with watershed post-processing to achieve superior performance for label-free cell segmentation across diverse cell types and imaging conditions.

## 🌟 Key Features

- **State-of-the-art Performance**: Outperforms existing methods like Cellpose, Stardist, and CALT-US on both test-set and zero-shot cross-dataset evaluation
- **Zero-shot Generalization**: Works on novel cell types and microscopes not seen during training
- **Distance Map Regression**: Predicts Euclidean distance to cell boundaries instead of binary masks, enabling better separation of densely packed cells
- **Vision Transformer Architecture**: Leverages SAM's ViT-based encoder pretrained on 11M images for robust feature extraction
- **Comprehensive Metrics**: Calculate 30+ morphological and intensity-based cell metrics
- **Easy Integration**: Simple Python API with minimal setup
- **Multiple Interfaces**: Command-line tool, Python API, GUI, and Napari plugin

## 📊 Performance

SAMCell demonstrates superior performance in both test-set and zero-shot cross-dataset evaluation:

### Test-Set Performance

| Dataset | Method | SEG | DET | OP_CSB |
|---------|--------|-----|-----|--------|
| LIVECell | **SAMCell** | **0.652** | **0.893** | **0.772** |
| | Cellpose | 0.589 | 0.779 | 0.684 |
| | Stardist | 0.572 | 0.771 | 0.671 |
| Cytoplasm | **SAMCell** | **0.611** | **0.866** | **0.739** |
| | Cellpose | 0.580 | 0.749 | 0.664 |
| | Stardist | 0.557 | 0.774 | 0.666 |

### Zero-Shot Cross-Dataset Performance

| Dataset | Method | SEG | DET | OP_CSB |
|---------|--------|-----|-----|--------|
| PBL-HEK | **SAMCell-Generalist** | **0.425** | **0.772** | **0.598** |
| | Cellpose-Cyto | 0.253 | 0.388 | 0.320 |
| | Stardist-Cyto | 0.142 | 0.236 | 0.189 |
| PBL-N2a | **SAMCell-Generalist** | **0.707** | **0.941** | **0.824** |
| | Cellpose-Cyto | 0.642 | 0.885 | 0.764 |
| | Stardist-Cyto | 0.597 | 0.851 | 0.724 |

*SAMCell-Generalist trained on LIVECell + Cytoplasm datasets. PBL-HEK and PBL-N2a contain novel cell types not seen during training.*

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI (recommended)
pip install samcell

# Or install from source
git clone https://github.com/saahilsanganeriya/SAMCell.git
cd SAMCell
pip install -e .
```

### Download Pre-trained Weights

Download the pre-trained SAMCell model weights:

```bash
# SAMCell-Generalist (recommended)
wget https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/samcell-generalist.pt

# Or SAMCell-Cyto
wget https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/samcell-cyto.pt
```

### Basic Usage

```python
import cv2
import samcell

# Load your microscopy image
image = cv2.imread('your_image.png', cv2.IMREAD_GRAYSCALE)

# Initialize SAMCell
model = samcell.FinetunedSAM('facebook/sam-vit-base')
model.load_weights('samcell-generalist.pt')

# Create pipeline
pipeline = samcell.SAMCellPipeline(model, device='cuda')

# Segment cells
labels = pipeline.run(image)

# Calculate metrics
metrics_df = pipeline.calculate_metrics(labels, image)
print(f"Found {len(metrics_df)} cells")

# Export results
pipeline.export_metrics(labels, 'cell_metrics.csv', image)
```

### Command Line Interface

```bash
# Basic segmentation
samcell segment image.png --model samcell-generalist.pt --output results/

# With comprehensive metrics
samcell segment image.png --model samcell-generalist.pt --output results/ --export-metrics

# Custom thresholds
samcell segment image.png --model samcell-generalist.pt --peak-threshold 0.5 --fill-threshold 0.1
```

## 📋 Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- transformers ≥ 4.26.0
- OpenCV ≥ 4.5.0
- scikit-image ≥ 0.19.0
- pandas ≥ 1.3.0

For GPU acceleration:
- CUDA-compatible GPU
- CUDA Toolkit ≥ 11.0

## 🔧 Advanced Usage

### Custom Thresholds

SAMCell uses two key thresholds for post-processing:

```python
# Default values (optimized across datasets)
pipeline = samcell.SAMCellPipeline(model, device='cuda')
labels = pipeline.run(image, cells_max=0.47, cell_fill=0.09)
```

### Batch Processing

```python
# Process multiple images
images = [cv2.imread(f'image_{i}.png', 0) for i in range(10)]

results = []
for image in images:
    labels = pipeline.run(image)
    metrics = pipeline.calculate_metrics(labels, image)
    results.append(metrics)

# Combine all metrics
import pandas as pd
all_metrics = pd.concat(results, ignore_index=True)
```

### Comprehensive Metrics

SAMCell calculates 30+ morphological and intensity metrics:

```python
# Basic metrics (fast)
basic_metrics = samcell.calculate_basic_metrics(labels, image)

# Include neighbor analysis
neighbor_metrics = samcell.calculate_neighbor_metrics(labels)

# Full analysis including texture (slower)
full_metrics = samcell.calculate_all_metrics(
    labels, image, include_texture=True
)
```

## 🖥️ GUI and Napari Plugin

### Standalone GUI

```bash
# Install GUI dependencies
pip install samcell[gui]

# Launch GUI
python -m samcell.gui
```

### Napari Plugin

```bash
# Install napari plugin
pip install samcell[napari]

# Launch napari and find SAMCell in the plugins menu
napari
```

## 📖 Documentation

### API Reference

#### `FinetunedSAM`

```python
model = samcell.FinetunedSAM(sam_model='facebook/sam-vit-base')
model.load_weights(weight_path, map_location='cuda')
```

#### `SAMCellPipeline`

```python
pipeline = samcell.SAMCellPipeline(
    model,                    # FinetunedSAM instance
    device='cuda',           # 'cuda' or 'cpu'
    crop_size=256,          # Patch size for sliding window
)

# Run segmentation
labels = pipeline.run(
    image,                   # Input grayscale image
    cells_max=0.47,         # Cell peak threshold
    cell_fill=0.09,         # Cell fill threshold
    return_dist_map=False   # Return distance map
)
```

#### Metrics Functions

```python
# Calculate all metrics
metrics_df = samcell.calculate_all_metrics(
    labels,                  # Segmentation labels
    original_image=None,     # Original image for intensity metrics
    include_texture=False,   # Include texture analysis
    neighbor_distance=10     # Distance for neighbor analysis
)

# Export to CSV
success = samcell.export_metrics_csv(
    labels,
    'output.csv',
    original_image=image,
    include_texture=False
)
```

### Available Metrics

SAMCell calculates comprehensive morphological metrics:

**Shape Metrics:**
- Area, Perimeter, Convex Area
- Compactness, Circularity, Roundness
- Aspect Ratio, Eccentricity, Solidity
- Major/Minor Axis Lengths

**Spatial Metrics:**
- Centroid coordinates
- Bounding box dimensions
- Number of neighbors
- Nearest neighbor distances

**Intensity Metrics** (when original image provided):
- Mean, Standard deviation, Min/Max intensity
- Intensity range and distribution

**Texture Metrics** (optional):
- GLCM-based features
- Contrast, Homogeneity, Energy
- Correlation, Dissimilarity

## 🔬 Method Overview

SAMCell introduces several key innovations for robust cell segmentation:

### 1. Distance Map Regression
Instead of predicting binary masks or multi-class segmentation, SAMCell predicts a continuous-valued distance map where each pixel value represents the normalized Euclidean distance from that pixel to its cell's boundary (0 to 1 range). This approach effectively addresses the challenge of segmenting densely packed cells with ambiguous boundaries.

### 2. Vision Transformer Architecture
SAMCell inherits SAM's ViT-based image encoder pretrained on 11 million diverse natural images. This extensive pretraining provides:
- Strong priors for boundary detection across imaging conditions
- Long-range dependency modeling via self-attention mechanisms
- Superior generalization to novel cell types and microscopes

### 3. Watershed Post-Processing
Converts predicted distance maps to discrete cell masks using the watershed algorithm:
- **Cell Peak Threshold (default: 0.47)**: Identifies cell centers from distance map peaks
- **Cell Fill Threshold (default: 0.09)**: Determines cell boundaries
- The watershed algorithm treats the distance map as a topographical surface, flooding from cell centers to naturally separate touching cells

### 4. Sliding Window Inference
Processes large microscopy images efficiently:
- Divides images into overlapping 256×256 patches (32-pixel overlap)
- Each patch upsampled to 1024×1024 for SAM's encoder
- Predictions stitched with cosine blending to avoid edge artifacts

### 5. No Prompting Required
Unlike vanilla SAM, SAMCell eliminates manual prompting by:
- Freezing the prompt encoder during fine-tuning
- Using SAM's default prompt embedding as a static input
- The mask decoder learns to predict distance maps from image embeddings alone

### Training Details
- **Model**: SAM-Base (ViT-B) with 89M parameters
- **Fine-tuning**: Full fine-tuning of image encoder and mask decoder
- **Loss**: Mean Squared Error (MSE) on sigmoid-activated predictions
- **Optimizer**: AdamW (lr=1e-4, weight decay=0.1)
- **Training**: Early stopping with patience=7, trained for 35 epochs on NVIDIA A100
- **Data Augmentation**: Random flip, rotation (-180° to 180°), scale (0.8-1.2×), brightness (0.95-1.05×), inversion
- **Preprocessing**: CLAHE (clipLimit=3.0, tileGridSize=8×8) for contrast enhancement

## 📊 Datasets

### Training Datasets
- **LIVECell**: 5,000+ phase-contrast images across 8 cell types containing ~1.7M individually annotated cells. All images captured with same microscope at standardized size (704×520 pixels). Provides large-scale training data with diverse cell morphologies and confluencies.
- **Cellpose Cytoplasm**: ~600 microscopy images from diverse internet sources. Includes both bright-field and fluorescent microscopy from different microscopes. Smaller but more diverse in imaging conditions. Images resized to 512×512 pixels preserving aspect ratio.

### Evaluation Datasets (Zero-Shot)
- **PBL-HEK**: 5 phase-contrast images of Human Embryonic Kidney 293 cells (~300 cells per image). Captured with different microscope than training data. Features densely packed cells with irregular morphologies.
- **PBL-N2a**: 5 phase-contrast images of Neuro-2a cells (~300 cells per image). Novel cell line and microscope not seen in training. More circular morphology with distinct boundaries compared to HEK cells.

Both evaluation datasets available at: https://github.com/saahilsanganeriya/SAMCell/releases/tag/v1

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 Citation

If you use SAMCell in your research, please cite our paper:

VandeLoo AD*, Malta NJ*, Sanganeriya S, Aponte E, van Zyl C, et al. (2025) SAMCell: Generalized label-free biological cell segmentation with segment anything. PLOS ONE 20(9): e0319532. https://doi.org/10.1371/journal.pone.0319532

```bibtex
@article{vandeloo2025samcell,
    title={SAMCell: Generalized label-free biological cell segmentation with segment anything},
    author={VandeLoo, Alexandra Dunnum and Malta, Nathan J and Sanganeriya, Saahil and Aponte, Emilio and van Zyl, Caitlin and Xu, Danfei and Forest, Craig},
    journal={PLOS ONE},
    volume={20},
    number={9},
    pages={e0319532},
    year={2025},
    publisher={Public Library of Science},
    doi={10.1371/journal.pone.0319532},
    url={https://doi.org/10.1371/journal.pone.0319532}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/saahilsanganeriya/SAMCell/issues)
- **Discussions**: [GitHub Discussions](https://github.com/saahilsanganeriya/SAMCell/discussions)
- **Email**: saahilsanganeriya@gatech.edu

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏛️ Institutions

This work was developed at:
- **Georgia Institute of Technology**
  - School of Biological Sciences
  - School of Computer Science  
  - Department of Biomedical Engineering
  - School of Mechanical Engineering
  - School of Interactive Computing

## 🙏 Acknowledgments

- Meta AI for the original Segment Anything Model
- The open-source community for tools and datasets
- Georgia Tech for computational resources
- All contributors and users of SAMCell

---

**SAMCell Team** - Making cell segmentation accessible to everyone! 🔬✨
