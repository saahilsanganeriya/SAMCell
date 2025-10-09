# SAMCell Evaluation Scripts

This directory contains scripts for evaluating SAMCell models and running ablation studies.

## Directory Structure

```
evaluation/
├── README.md                   # This file
├── evaluate_model.py           # Single model evaluation script
├── evaluate_checkpoints.py     # Batch evaluation of wandb checkpoints
└── ablations/                  # Ablation study scripts
    ├── README.md
    ├── run_ablations.py        # Main ablation study runner
    └── visualize_ablations.py  # Visualization script
```

## Dependencies

```bash
pip install stardist  # For IoU-based metrics
pip install pandas matplotlib seaborn  # For analysis and visualization
```

## Scripts

### `evaluate_model.py`

Evaluate a single model on a dataset.

```bash
python evaluate_model.py \
    --model path/to/model.pt \
    --dataset path/to/dataset \
    --output results.csv
```

### `evaluate_checkpoints.py`

Batch evaluate multiple checkpoints from WandB.

```bash
python evaluate_checkpoints.py \
    --run-id abc123 \
    --datasets path/to/dataset1 path/to/dataset2 \
    --output eval_results.csv
```

### Ablation Studies

See `ablations/README.md` for details on running ablation studies.

## Evaluation Metrics

The evaluation scripts calculate:

### Cell Tracking Challenge (CTC) Metrics
- **SEG**: Segmentation accuracy
- **DET**: Detection accuracy
- **CSB**: Combined score (SEG + DET) / 2

### StarDist Metrics (at multiple IoU thresholds: 0.50-0.90)
- **Precision**: Ratio of correctly detected cells to all detected cells
- **Recall**: Ratio of correctly detected cells to all ground truth cells
- **F1**: Harmonic mean of precision and recall
- **Accuracy**: Overall accuracy
- **Panoptic Quality**: Combined segmentation and detection quality

## Dataset Format

Evaluation datasets should contain:
- `imgs.npy`: Image array (N, H, W) or (N, H, W, C)
- `anns.npy`: Annotation masks (N, H, W) - labeled instances

## Output Format

Results are saved as CSV files with columns:
- `model_name`: Name or ID of the model
- `dataset`: Dataset name
- `seg_value`: SEG score
- `det_value`: DET score
- `csb_value`: CSB score
- `precision`, `recall`, `f1`, `accuracy`, `panoptic_quality`: StarDist metrics
- Additional columns for metrics at different IoU thresholds

## Cell Tracking Challenge Binaries

For CTC metrics, you need the evaluation binaries:
1. Download from: https://celltrackingchallenge.net/evaluation-methodology/
2. Place `SEGMeasure` and `DETMeasure` in `cell-tracking-binaries/` directory
3. Make executable: `chmod +x cell-tracking-binaries/*`
