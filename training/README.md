# SAMCell Training Scripts

This directory contains training scripts for SAMCell models.

## Scripts

### `train.py` - Comprehensive Training Script

Full-featured training script with:
- Multi-dataset support (concatenates multiple datasets)
- **Fine-tuning from pretrained weights** (NEW!)
- Early stopping
- Mixed precision training (AMP)
- Comprehensive WandB logging (gradients, predictions, system metrics)
- Model checkpointing (periodic + best model)
- Command-line arguments

**Usage:**

```bash
# Train on single dataset
python train.py --datasets /path/to/dataset1 --batch-size 4 --num-epochs 40

# Train on multiple datasets
python train.py --datasets /path/to/dataset1 /path/to/dataset2 \
                --batch-size 8 \
                --sam-model facebook/sam-vit-base \
                --learning-rate 1e-4 \
                --num-epochs 40

# Train with early stopping
python train.py --datasets /path/to/dataset1 \
                --patience 7 \
                --min-delta 1e-4

# Train without WandB
python train.py --datasets /path/to/dataset1 --no-wandb

# Use SAM Large
python train.py --datasets /path/to/dataset1 \
                --sam-model facebook/sam-vit-large

# Fine-tune from pretrained weights
python train.py --datasets /path/to/dataset1 \
                --pretrained-weights /path/to/checkpoint.pt \
                --learning-rate 5e-5 \
                --num-epochs 20
```

**Arguments:**

- `--datasets`: List of dataset paths (required). Each should contain `imgs.npy`, `dist_maps.npy`, and optionally `wms.npy`
- `--sam-model`: SAM variant (`facebook/sam-vit-base`, `facebook/sam-vit-large`, `facebook/sam-vit-huge`)
- `--pretrained-weights`: Path to pretrained SAMCell checkpoint (.pt file) to start training from (optional)
- `--batch-size`: Batch size (default: 4)
- `--num-epochs`: Number of epochs (default: 40)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 0.1)
- `--patience`: Early stopping patience (default: 7)
- `--min-delta`: Minimum loss improvement for early stopping (default: 1e-4)
- `--output-dir`: Checkpoint output directory (default: ../checkpoints)
- `--save-interval`: Save checkpoint every N epochs (default: 5)
- `--finetune-vision`: Fine-tune vision encoder (default: True)
- `--no-finetune-vision`: Freeze vision encoder
- `--no-wandb`: Disable WandB logging

### Fine-tuning from Pretrained Weights

You can start training from a pretrained SAMCell checkpoint instead of from scratch:

```bash
# Fine-tune from samcell-generalist.pt
python train.py --datasets /path/to/dataset1 \
                --pretrained-weights ../samcell-generalist.pt \
                --learning-rate 5e-5 \
                --num-epochs 20 \
                --patience 7

# Fine-tune from your own checkpoint
python train.py --datasets /path/to/dataset1 \
                --pretrained-weights ../checkpoints/my-model/best_model.pt \
                --learning-rate 5e-5 \
                --num-epochs 20
```

**Benefits of fine-tuning:**
- Faster convergence (typically 2x faster)
- Better final performance
- Requires fewer epochs (20-25 vs 40-50)
- More stable training with lower learning rate

**Recommended settings for fine-tuning:**
- Learning rate: `5e-5` (half of from-scratch)
- Epochs: `20-25` (half of from-scratch)
- Batch size: Can often use larger (4-8)
- Patience: `5-7` (converges faster)

### `train_simple.py` - Simple Training Script

Minimal training script for quick experiments:
- Hardcoded paths and parameters
- Basic WandB logging
- Simpler to understand and modify

**Usage:**

Edit the script to set your dataset paths and parameters, then:

```bash
python train_simple.py
```

## Dataset Format

Datasets should contain:
- `imgs.npy`: Image array (N, H, W) or (N, H, W, C)
- `dist_maps.npy`: Distance map ground truth (N, H, W)
- `wms.npy`: (Optional) Weight maps for loss weighting (N, H, W)

## Model Checkpoints

Models are saved to `<output-dir>/samcell-<datasets>-<model>/`:
- `checkpoint_epoch_N.pt`: Periodic checkpoints
- `best_model.pt`: Best model based on training loss
- `final_model.pt`: Final model after all epochs

## WandB Logging

When WandB is enabled, the following are logged:
- **Training metrics**: Step loss, learning rate, epoch, parameter norm
- **System metrics**: CPU/memory usage, GPU memory
- **Gradients**: Norm, mean, std (every 100 steps)
- **Predictions**: Visual comparison of predictions vs ground truth (every 100 steps)
- **Epoch statistics**: Mean/std/min/max loss per epoch
- **Model artifacts**: Checkpoints uploaded to WandB

Project name: "SAMCell"

Run naming format: `{datasets}-{model}-bs{batch_size}-lr{learning_rate}`
