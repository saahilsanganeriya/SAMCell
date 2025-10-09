# SAMCell Dataset Processing Framework

Unified framework for converting various microscopy dataset formats into SAMCell training format.

## Overview

This framework provides utilities to process different dataset types into the standardized format required by SAMCell's training pipeline:

- **imgs.npy**: Preprocessed images (N, H, W, 3) float32, CLAHE-normalized BGR
- **dist_maps.npy**: Distance maps (N, H, W) float32, normalized to [0, 1]
- **wms.npy**: Weight maps (N, H, W) float32 (optional, for boundary emphasis)

## Supported Dataset Types

1. **LIVECell**: COCO-format phase-contrast microscopy dataset
2. **Cellpose**: Numbered image and mask pairs
3. **Custom**: Any dataset with separate image and mask folders

## Files

- `process_dataset.py`: Main processing script with CLI
- `dataset_utils.py`: Shared utility functions for all dataset types

## Quick Start

### 1. Install Dependencies

```bash
# Required
pip install numpy opencv-python-headless

# Optional (recommended)
pip install scipy tqdm matplotlib

# For LIVECell only
pip install pycocotools
```

### 2. Process Your Dataset

#### LIVECell Dataset

```bash
# Full training set
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train \
    --split train

# 50% subset (recommended for faster training)
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train50 \
    --split train_50pct

# Validation or test set
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-val \
    --split val
```

#### Cellpose Dataset

```bash
python process_dataset.py cellpose \
    --input /path/to/CellPose/train_raw \
    --output ../datasets/Cellpose-train \
    --target-size 512
```

**File naming convention:**
- Images: `000_img.png`, `001_img.png`, `002_img.png`, ...
- Masks: `000_masks.png`, `001_masks.png`, `002_masks.png`, ...

#### Custom Dataset

```bash
python process_dataset.py custom \
    --images /path/to/images \
    --masks /path/to/masks \
    --output ../datasets/MyDataset \
    --target-size 512 \
    --image-ext .tif \
    --mask-suffix _seg
```

**Pairing convention:**
- Image: `cell_001.tif` → Mask: `cell_001_seg.tif`
- Image: `sample.png` → Mask: `sample_mask.png` (default suffix)

### 3. Train SAMCell

After processing, train with the standard training script:

```bash
cd ../training
python train.py --datasets ../datasets/MyDataset --batch-size 4 --num-epochs 40
```

## Command Reference

### Common Options (All Dataset Types)

| Option | Description | Default |
|--------|-------------|---------|
| `--output`, `-o` | Output directory for processed dataset | Required |
| `--create-weights` | Create weight maps (emphasizes cell boundaries) | False |
| `--visualize` | Generate visualization of random samples | False |

### LIVECell Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Path to LIVECell_dataset_2021 folder | Required |
| `--split` | Dataset split (see below) | `train_50pct` |

**Available splits:**
- `train`: Full training set (~5000 images)
- `train_50pct`: 50% of training set (~2500 images) **← Recommended**
- `train_25pct`: 25% of training set (~1250 images)
- `train_5pct`, `train_4pct`, `train_2pct`: Smaller subsets
- `val`: Validation set
- `test`: Test set

### Cellpose Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Folder containing numbered images/masks | Required |
| `--img-pattern` | Image filename pattern | `{:03d}_img.png` |
| `--mask-pattern` | Mask filename pattern | `{:03d}_masks.png` |
| `--target-size` | Resize to this size (maintains aspect ratio) | `512` |

### Custom Dataset Specific Options

| Option | Description | Default |
|--------|-------------|---------|
| `--images` | Folder containing images | Required |
| `--masks` | Folder containing masks | Required |
| `--image-ext` | Image file extension | `.png` |
| `--mask-suffix` | Suffix for mask filenames | `_mask` |
| `--target-size` | Resize to this size (maintains aspect ratio) | `512` |

## Processing Details

### Preprocessing Pipeline

1. **Grayscale Conversion**: Multi-channel images converted to grayscale
2. **CLAHE Normalization**: Contrast Limited Adaptive Histogram Equalization
   - `clipLimit=3.0`
   - `tileGridSize=(8, 8)`
3. **BGR Conversion**: Convert to 3-channel BGR for SAM processor compatibility
4. **Normalization**: Scale to [0, 1] float32 range

### Distance Map Creation

Distance maps represent the normalized Euclidean distance from each pixel to the nearest cell boundary:
- **0.0**: Background or cell edge
- **1.0**: Maximum distance from boundary (cell center)

This representation helps SAMCell better separate densely packed cells.

### Weight Maps (Optional)

Weight maps increase loss emphasis at cell boundaries to help with:
- Separating touching cells
- Accurate boundary detection
- Handling crowded regions

Generated using morphological operations + Gaussian smoothing.

### Resizing Strategy

For Cellpose and Custom datasets:
1. Resize longest dimension to `target_size`
2. Maintain aspect ratio
3. Pad shorter dimension with zeros to make square
4. Uses `INTER_LINEAR` for images, `INTER_NEAREST` for masks

## Output Format

After processing, your output directory will contain:

```
output_dir/
├── imgs.npy           # Preprocessed images (N, H, W, 3) float32
├── dist_maps.npy      # Distance maps (N, H, W) float32
└── wms.npy            # Weight maps (optional) (N, H, W) float32
```

These files are directly compatible with `SAMDataset` in `samcell/dataset_livecell.py`.

## Examples

### Example 1: Train on LIVECell + Cellpose (Combined)

```bash
# Process LIVECell
python process_dataset.py livecell \
    --input /data/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train50 \
    --split train_50pct \
    --visualize

# Process Cellpose
python process_dataset.py cellpose \
    --input /data/CellPose/train_raw \
    --output ../datasets/Cellpose-train \
    --target-size 512 \
    --visualize

# Train on both datasets
cd ../training
python train.py \
    --datasets ../datasets/LIVECell-train50 ../datasets/Cellpose-train \
    --batch-size 8 \
    --num-epochs 40 \
    --sam-model facebook/sam-vit-base
```

### Example 2: Process Custom TIF Dataset

```bash
# Your data structure:
# /data/my_cells/images/sample_001.tif
# /data/my_cells/masks/sample_001_segmentation.tif

python process_dataset.py custom \
    --images /data/my_cells/images \
    --masks /data/my_cells/masks \
    --output ../datasets/MyCells \
    --image-ext .tif \
    --mask-suffix _segmentation \
    --target-size 512 \
    --create-weights \
    --visualize
```

### Example 3: Create Small Test Dataset

```bash
# Process small subset for testing
python process_dataset.py livecell \
    --input /data/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-test \
    --split train_2pct \
    --visualize

# Quick training test
cd ../training
python train_simple.py  # Edit to point to ../datasets/LIVECell-test
```

## Troubleshooting

### Issue: "No images found"

**Solution**: Check file paths and extensions
```bash
# List files to verify
ls /path/to/images/
# Update --image-ext if needed
python process_dataset.py custom ... --image-ext .tif
```

### Issue: "Dimension mismatch"

**Solution**: Images and masks must have same dimensions before resizing
```python
# Check dimensions with:
import cv2
img = cv2.imread('image.png')
mask = cv2.imread('mask.png', cv2.IMREAD_UNCHANGED)
print(f"Image: {img.shape}, Mask: {mask.shape}")
```

### Issue: "pycocotools not found" (LIVECell only)

**Solution**:
```bash
pip install pycocotools
```

### Issue: Processed images look too dark/bright

**Solution**: The CLAHE preprocessing is automatic. If you need different parameters, modify `dataset_utils.py`:
```python
# In preprocess_image():
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))  # Adjust these
```

### Issue: Out of memory during processing

**Solution**: Process in batches or reduce image size
```bash
# Reduce target size
python process_dataset.py cellpose ... --target-size 256
```

## Integration with Training

The output format is directly compatible with SAMCell training scripts:

### Using train.py (Comprehensive)

```bash
python train.py \
    --datasets ../datasets/Dataset1 ../datasets/Dataset2 \
    --batch-size 4 \
    --num-epochs 40 \
    --learning-rate 1e-4 \
    --patience 7
```

### Using train_simple.py

Edit `train_simple.py` to set:
```python
dataset_path = '../datasets/YourDataset/'
img_path = dataset_path + 'imgs.npy'
ann_path = dataset_path + 'dist_maps.npy'
weight_path = dataset_path + 'wms.npy'  # or None
```

## Advanced Usage

### Programmatic Use

You can also import and use the processing functions directly:

```python
from dataset_utils import (
    preprocess_image,
    create_distance_map,
    save_dataset
)

# Process single image
import cv2
img = cv2.imread('cell.png')
mask = cv2.imread('cell_mask.png', cv2.IMREAD_UNCHANGED)

# Preprocess
img_processed = preprocess_image(img)
dist_map = create_distance_map(mask)

# Or process entire dataset
from process_dataset import process_cellpose

imgs, dist_maps, weight_maps = process_cellpose(
    input_folder='/path/to/data',
    output_dir='../datasets/output',
    target_size=512,
    create_weights=True
)
```

### Custom Preprocessing

To add custom preprocessing, modify `dataset_utils.py`:

```python
def preprocess_image(img):
    # Your custom preprocessing here
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply your custom normalization
    img_norm = your_custom_function(img_gray)

    # Convert to BGR for SAM
    img_bgr = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    return img_bgr
```

## Dataset Statistics

After processing, the script will display:
- Number of samples
- Image and distance map shapes
- Value ranges
- Weight map statistics (if generated)

Example output:
```
Dataset Statistics:
============================================================
Number of samples: 523
Image shape: (523, 512, 512, 3)
Distance map shape: (523, 512, 512)
Image range: [0.000, 1.000]
Distance map range: [0.000, 1.000]
============================================================
```

## Visualization

Use `--visualize` to generate sample images showing:
1. Preprocessed image (with CLAHE)
2. Distance map (color-coded)
3. Overlay (image + distance map)

Saved to `output_dir/sample_*.png`

## Performance Tips

1. **Use `scipy`**: Much faster distance transform than OpenCV
   ```bash
   pip install scipy
   ```

2. **Smaller target size**: Faster processing and training
   ```bash
   --target-size 256  # Instead of 512
   ```

3. **Skip weight maps**: If not needed for training
   ```bash
   # Don't use --create-weights
   ```

4. **Use train_50pct for LIVECell**: Good performance without full dataset
   ```bash
   --split train_50pct
   ```

## Citation

If you use this framework, please cite the SAMCell paper:

```
VandeLoo AD*, Malta NJ*, Sanganeriya S, Aponte E, van Zyl C, et al. (2025)
SAMCell: Generalized label-free biological cell segmentation with segment anything.
PLOS ONE 20(9): e0319532. https://doi.org/10.1371/journal.pone.0319532
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/saahilsanganeriya/SAMCell/issues
- Email: saahilsanganeriya@gatech.edu
