# Ablation Studies

This directory contains scripts for running systematic ablation studies on SAMCell.

## Studies Included

The ablation framework supports multiple studies:

1. **Patch Size Ablation**: Test different crop sizes (128, 256, 512)
2. **Dataset Combination Ablation**: Compare training on different dataset combinations
3. **Model Architecture Ablation**: Compare SAM-ViT-Base vs SAM-ViT-Large
4. **Pretraining Ablation**: Compare pretrained weights vs random initialization
5. **Per-Epoch Analysis**: Track performance evolution during training

## Usage

### Run Specific Study

```bash
# Run only patch size ablation
python run_ablations.py --study 1 --datasets PBL_HEK,PBL_N2A

# Run only model architecture comparison
python run_ablations.py --study 3 --patch-size 256 --datasets PBL_HEK

# Run per-epoch analysis
python run_ablations.py --study 5 --patch-size 256
```

### Run All Studies

```bash
python run_ablations.py --datasets PBL_HEK,PBL_N2A
```

## Configuration

Edit `run_ablations.py` to configure:
- Dataset paths
- WandB run IDs for each study
- Threshold values (cells_max, cell_fill)
- Path to CTC evaluation binaries

## Output

Results are saved to `ablation_results/`:
- `study_1_patch_sizes.csv`: Patch size comparison results
- `study_2_dataset_combos.csv`: Dataset combination results
- `study_3_sam_models.csv`: Model architecture comparison
- `study_4_pretraining.csv`: Pretraining comparison
- `study_5_per_epoch.csv`: Per-epoch performance
- `segmentations/`: Cached segmentation masks (for Study 5)

## Visualization

After running ablations:

```bash
python visualize_ablations.py --results-dir ablation_results/
```

Generates plots for each study in `ablation_results/figures/`.

## WandB Integration

The scripts automatically:
- Download checkpoints from WandB
- Cache segmentation results to avoid redundant inference
- Log progress and errors

Make sure you're logged in to WandB:
```bash
wandb login
```

## Parallelization

The ablation runner uses ThreadPoolExecutor for parallel evaluation. Adjust `max_workers` in the script to control parallelism based on your hardware.

## Error Handling

- Failed evaluations are retried up to 2 times
- Results are saved incrementally (every 100 evaluations)
- Logs are saved to `logs/ablation_studies_<timestamp>.log`
