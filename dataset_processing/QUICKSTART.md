# Quick Start Guide

## TL;DR

```bash
# Install dependencies
pip install numpy opencv-python-headless scipy tqdm pycocotools

# Process LIVECell (recommended for most users)
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train50 \
    --split train_50pct

# Process Cellpose
python process_dataset.py cellpose \
    --input /path/to/CellPose/train_raw \
    --output ../datasets/Cellpose-train

# Train SAMCell
cd ../training
python train.py --datasets ../datasets/LIVECell-train50 --batch-size 4 --num-epochs 40
```

## What This Does

Converts your microscopy images + masks into SAMCell training format:
1. **Preprocesses images**: CLAHE normalization, grayscale→BGR conversion
2. **Creates distance maps**: Converts segmentation masks to distance fields
3. **Saves as .npy files**: Ready for `train.py` or `train_simple.py`

## Dataset Types

### LIVECell (COCO Format)
- Large phase-contrast dataset
- 8 cell lines, ~5000 images
- Use `--split train_50pct` for faster training

### Cellpose (Numbered Pairs)
- Image: `000_img.png`, `001_img.png`, ...
- Mask: `000_masks.png`, `001_masks.png`, ...
- Automatically resized to 512×512

### Custom (Any Format)
- Separate image and mask folders
- Flexible naming: `cell_001.tif` + `cell_001_mask.tif`

## Output Files

Your `--output` directory will contain:
- `imgs.npy`: Preprocessed images (N, H, W, 3)
- `dist_maps.npy`: Distance maps (N, H, W)
- `wms.npy`: Weight maps (optional, use `--create-weights`)

## Common Commands

```bash
# LIVECell full training set
python process_dataset.py livecell -i /data/LIVECell_dataset_2021 -o ../datasets/LIVECell-train --split train

# LIVECell 50% (recommended)
python process_dataset.py livecell -i /data/LIVECell_dataset_2021 -o ../datasets/LIVECell-train50 --split train_50pct

# Cellpose with visualization
python process_dataset.py cellpose -i /data/CellPose/train_raw -o ../datasets/Cellpose-train --visualize

# Custom dataset (TIF files)
python process_dataset.py custom \
    --images /data/images \
    --masks /data/masks \
    -o ../datasets/MyData \
    --image-ext .tif \
    --mask-suffix _seg

# Custom with weight maps
python process_dataset.py custom \
    --images /data/images \
    --masks /data/masks \
    -o ../datasets/MyData \
    --create-weights
```

## Training After Processing

### Single Dataset
```bash
python train.py --datasets ../datasets/MyDataset --batch-size 4 --num-epochs 40
```

### Multiple Datasets (Combined)
```bash
python train.py \
    --datasets ../datasets/LIVECell-train50 ../datasets/Cellpose-train \
    --batch-size 8 \
    --num-epochs 40
```

### Simple Training (Edit Paths in Script)
```bash
# Edit train_simple.py:
#   dataset_path = '../datasets/MyDataset/'
python train_simple.py
```

## Troubleshooting

**"No images found"**: Check paths and use `--image-ext .tif` if not PNG

**"pycocotools not found"**: `pip install pycocotools` (LIVECell only)

**Out of memory**: Use `--target-size 256` or smaller dataset split

**Images too dark**: CLAHE is automatic, working as intended

## Need More Details?

See `README.md` for complete documentation.
