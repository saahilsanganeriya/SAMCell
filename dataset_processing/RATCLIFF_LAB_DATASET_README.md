# Ratcliff Lab Dataset Processing for SAMCell

## ✅ Datasets Created Successfully!

Two datasets have been prepared from your Cellpose segmentations for SAMCell training:

### Dataset 1: RatcliffLab-model
- **Source**: `images_for_cellpose_model/`
- **Samples**: 9 images
- **Location**: `SAMCell/datasets/RatcliffLab-model/`
- **Files**:
  - `imgs.npy` (9.0 MB) - Green channel images, normalized 0-1
  - `dist_maps.npy` (9.0 MB) - Distance transform maps from masks

### Dataset 2: RatcliffLab-analyze
- **Source**: `images_to_analyze/`
- **Samples**: 12 images
- **Location**: `SAMCell/datasets/RatcliffLab-analyze/`
- **Files**:
  - `imgs.npy` (12 MB) - Green channel images, normalized 0-1
  - `dist_maps.npy` (12 MB) - Distance transform maps from masks

---

## 📊 Dataset Specifications

Both datasets:
- **Image size**: 512 × 512 pixels (resized with padding)
- **Image format**: Float32, normalized to [0, 1]
- **Distance maps**: Float32, normalized to [0, 1]
- **Channel**: Green fluorescence (GFP) only
- **Quality**: Maintains aspect ratio, zero-padding added

---

## 🚀 How to Train SAMCell

### Train on Single Dataset

```bash
cd SAMCell/training

# Train on RatcliffLab-model (9 images)
python train.py --datasets ../datasets/RatcliffLab-model \
                --batch-size 2 \
                --num-epochs 40 \
                --learning-rate 1e-4

# Train on RatcliffLab-analyze (12 images)
python train.py --datasets ../datasets/RatcliffLab-analyze \
                --batch-size 2 \
                --num-epochs 40 \
                --learning-rate 1e-4
```

### Train on Both Datasets Combined

```bash
cd SAMCell/training

# Train on both datasets (21 images total)
python train.py --datasets ../datasets/RatcliffLab-model ../datasets/RatcliffLab-analyze \
                --batch-size 4 \
                --num-epochs 40 \
                --learning-rate 1e-4 \
                --sam-model facebook/sam-vit-base
```

### Training with Different SAM Models

```bash
# SAM Base (default, ~90M parameters)
python train.py --datasets ../datasets/RatcliffLab-model --sam-model facebook/sam-vit-base

# SAM Large (~300M parameters)
python train.py --datasets ../datasets/RatcliffLab-model --sam-model facebook/sam-vit-large

# SAM Huge (~600M parameters)
python train.py --datasets ../datasets/RatcliffLab-model --sam-model facebook/sam-vit-huge
```

### Advanced Training Options

```bash
# With early stopping
python train.py --datasets ../datasets/RatcliffLab-model \
                --patience 7 \
                --min-delta 1e-4

# Without WandB logging
python train.py --datasets ../datasets/RatcliffLab-model --no-wandb

# Custom checkpoint directory
python train.py --datasets ../datasets/RatcliffLab-model \
                --output-dir ./my_checkpoints

# Freeze vision encoder (faster training)
python train.py --datasets ../datasets/RatcliffLab-model --no-finetune-vision
```

---

## 🔄 Re-process Datasets

If you need to re-process with different settings:

```bash
cd SAMCell/dataset_processing

# Standard 512x512
python process_ratcliff_lab_data.py \
    --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze" \
    --output-dir ../datasets/RatcliffLab-analyze

# Larger images (1024x1024)
python process_ratcliff_lab_data.py \
    --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze" \
    --output-dir ../datasets/RatcliffLab-analyze-1024 \
    --target-size 1024

# With weight maps for weighted loss
python process_ratcliff_lab_data.py \
    --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze" \
    --output-dir ../datasets/RatcliffLab-analyze-weighted \
    --create-weights
```

---

## 📝 What the Processing Does

The `process_ratcliff_lab_data.py` script:

1. **Loads green channel images** from `exported_images_ch1/`
2. **Loads segmentation masks** from `exported_masks_raw/`
3. **Resizes to 512×512**:
   - Maintains aspect ratio
   - Pads with zeros to make square
4. **Creates distance maps**:
   - Computes distance transform from cell boundaries
   - Normalizes to 0-1 range
5. **Normalizes images** to 0-1 range
6. **Saves as numpy arrays**:
   - `imgs.npy`: (N, 512, 512) float32
   - `dist_maps.npy`: (N, 512, 512) float32
   - `wms.npy`: (N, 512, 512) float32 (if --create-weights)

---

## 🎯 Training Recommendations

### For Small Datasets (9-21 images):

1. **Use smaller batch size**: 2-4
2. **More epochs**: 40-60
3. **Enable early stopping**: `--patience 7`
4. **Consider freezing vision encoder**: `--no-finetune-vision` (faster, less overfitting)
5. **Use data augmentation** if available

### For Combined Training:

```bash
# Recommended settings for your 21 combined images
python train.py \
    --datasets ../datasets/RatcliffLab-model ../datasets/RatcliffLab-analyze \
    --batch-size 4 \
    --num-epochs 50 \
    --learning-rate 1e-4 \
    --patience 10 \
    --sam-model facebook/sam-vit-base \
    --save-interval 5
```

---

## 📦 Output Structure

After training, checkpoints will be saved to:

```
SAMCell/checkpoints/samcell-RatcliffLab-model+RatcliffLab-analyze-sam-vit-base/
├── checkpoint_epoch_0.pt
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── ...
├── best_model.pt          ← Best model based on loss
└── final_model.pt         ← Final model after training
```

---

## 🔍 Monitoring Training

If WandB is enabled (default), view training at:
- Project: "SAMCell"
- Run name format: `RatcliffLab-model+RatcliffLab-analyze-sam-vit-base-bs4-lr1.0e-04`

Logged metrics:
- Training loss per step
- Learning rate schedule
- Gradient statistics
- Prediction visualizations
- Epoch statistics

---

## ✨ Next Steps

1. **Start with single dataset**:
   ```bash
   cd SAMCell/training
   python train.py --datasets ../datasets/RatcliffLab-analyze --batch-size 2 --num-epochs 40
   ```

2. **Monitor training** via WandB or console output

3. **Evaluate** on test images (if you have them)

4. **Fine-tune** with both datasets combined for better generalization

5. **Use trained model** for inference on new mitotic spindle images

---

## 📚 Dataset Format Details

### imgs.npy
- **Shape**: (N, H, W) where N=number of images, H=W=512
- **Dtype**: float32
- **Range**: [0.0, 1.0]
- **Content**: Green fluorescence (GFP) channel, normalized

### dist_maps.npy
- **Shape**: (N, H, W) where N=number of images, H=W=512
- **Dtype**: float32
- **Range**: [0.0, 1.0]
- **Content**: Distance transform from cell boundaries
  - 0.0 at cell boundaries
  - Higher values toward cell centers
  - Normalized to 0-1

### wms.npy (optional)
- **Shape**: (N, H, W) where N=number of images, H=W=512
- **Dtype**: float32
- **Range**: Variable
- **Content**: Weight maps emphasizing cell boundaries

---

## 🛠️ Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch-size 1` or `--batch-size 2`
- Use smaller model: `--sam-model facebook/sam-vit-base`

### Training too slow
- Freeze vision encoder: `--no-finetune-vision`
- Reduce image size: `--target-size 256` when processing

### Not improving
- Try more epochs: `--num-epochs 60`
- Adjust learning rate: `--learning-rate 5e-5`
- Combine both datasets for more training data

---

**Created**: October 6, 2025  
**Script**: `process_ratcliff_lab_data.py`  
**Source data**: Ratcliff Lab mitotic spindle segmentation with Cellpose

