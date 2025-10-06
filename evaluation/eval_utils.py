"""
Shared evaluation utilities for SAMCell.

This module contains common functions used by evaluation and ablation scripts.
"""

import os
import subprocess
import re
import numpy as np
import cv2
import shutil
from pathlib import Path

# Optional dependencies
try:
    from stardist.matching import matching_dataset
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False
    print("Warning: stardist not installed. StarDist metrics will not be available.")


def run_evaluation(binary, eval_folder, binary_path, sequence, num_digits):
    """Run a Cell Tracking Challenge evaluation binary.

    Parameters
    ----------
    binary : str
        Name of the binary (e.g., 'SEGMeasure', 'DETMeasure')
    eval_folder : str
        Path to evaluation folder
    binary_path : str
        Path to directory containing binaries
    sequence : str
        Sequence number (e.g., '01')
    num_digits : str
        Number of digits in filenames (e.g., '04')

    Returns
    -------
    float or None
        Evaluation score, or None if parsing failed
    """
    binary_full_path = os.path.join(binary_path, binary)

    # Make sure binary is executable
    try:
        os.chmod(binary_full_path, 0o755)
    except:
        pass

    result = subprocess.run(
        [binary_full_path, eval_folder, sequence, num_digits],
        capture_output=True,
        text=True
    )

    output = result.stdout.strip()
    match = re.search(r"([0-9]+\.[0-9]+)", output)

    if match:
        return float(match.group(1))
    else:
        print(f"Error parsing output from {binary}: {output}")
        return None


def create_evaluation_structure(base_path, sequence_number=1, num_digits=4):
    """Create folder structure needed for Cell Tracking Challenge evaluation.

    Parameters
    ----------
    base_path : str
        Base path for evaluation
    sequence_number : int
        Sequence number (default: 1)
    num_digits : int
        Number of digits in filenames (default: 4)

    Returns
    -------
    tuple
        (gt_seg_dir, gt_tra_dir, res_dir)
    """
    seq_str = f"{sequence_number:02d}"
    gt_seg_dir = os.path.join(base_path, f"{seq_str}_GT", "SEG")
    gt_tra_dir = os.path.join(base_path, f"{seq_str}_GT", "TRA")
    res_dir = os.path.join(base_path, f"{seq_str}_RES")

    os.makedirs(gt_seg_dir, exist_ok=True)
    os.makedirs(gt_tra_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    return gt_seg_dir, gt_tra_dir, res_dir


def add_to_cell_tracking_challenge_format(gt_data, pred_data, gt_seg_dir, gt_tra_dir, res_dir):
    """Save ground truth and predicted segmentation masks in CTC format.

    Parameters
    ----------
    gt_data : np.ndarray
        Ground truth masks
    pred_data : np.ndarray
        Predicted masks
    gt_seg_dir : str
        Ground truth segmentation directory
    gt_tra_dir : str
        Ground truth tracking directory
    res_dir : str
        Results directory
    """
    # Handle ground truth data of shape (N, 1, H, W)
    if len(gt_data.shape) == 4 and gt_data.shape[1] == 1:
        gt_data = gt_data[:, 0, :, :]

    # Handle prediction data of shape (N, 1, H, W)
    if len(pred_data.shape) == 4 and pred_data.shape[1] == 1:
        pred_data = pred_data[:, 0, :, :]

    for i in range(len(gt_data)):
        filename_gt_seg = os.path.join(gt_seg_dir, f"man_seg{i:04d}.tif")
        filename_gt_tra = os.path.join(gt_tra_dir, f"man_track{i:04d}.tif")
        filename_pred = os.path.join(res_dir, f"mask{i:04d}.tif")

        cv2.imwrite(filename_gt_seg, gt_data[i].astype(np.uint16))
        cv2.imwrite(filename_gt_tra, gt_data[i].astype(np.uint16))
        cv2.imwrite(filename_pred, pred_data[i].astype(np.uint16))


def calculate_stardist_metrics(gt_data, pred_data):
    """Calculate metrics using stardist's matching_dataset function.

    Parameters
    ----------
    gt_data : np.ndarray
        Ground truth masks
    pred_data : np.ndarray
        Predicted masks

    Returns
    -------
    dict
        Dictionary containing metrics at various IoU thresholds
    """
    if not STARDIST_AVAILABLE:
        print("Warning: stardist not installed, skipping StarDist metrics")
        return {}

    # Make sure the ground truth and predictions are int32
    gt_data = gt_data.astype(np.int32)
    pred_data = pred_data.astype(np.int32)

    # Handle different array shapes for ground truth
    if len(gt_data.shape) > 3:
        if len(gt_data.shape) == 4 and gt_data.shape[1] == 1:
            gt_data = gt_data[:, 0, :, :]

    # Handle different array shapes for predictions
    if len(pred_data.shape) > 3:
        if len(pred_data.shape) == 4 and pred_data.shape[1] == 1:
            pred_data = pred_data[:, 0, :, :]

    # IoU thresholds
    taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    # Calculate metrics at different IoU thresholds
    stats = [matching_dataset(gt_data, pred_data, thresh=t, show_progress=False, parallel=True) for t in taus]

    # Create a dictionary with metrics
    results = {}
    for i, tau in enumerate(taus):
        stat = stats[i]
        results[f"iou_{tau}"] = {
            "precision": stat.precision,
            "recall": stat.recall,
            "f1": stat.f1,
            "accuracy": stat.accuracy,
            "panoptic_quality": stat.panoptic_quality,
            "mean_true_score": stat.mean_true_score,
            "mean_matched_score": stat.mean_matched_score
        }

    # Also add default metrics at IoU 0.5
    results["precision"] = stats[0].precision
    results["recall"] = stats[0].recall
    results["f1"] = stats[0].f1
    results["accuracy"] = stats[0].accuracy
    results["panoptic_quality"] = stats[0].panoptic_quality

    return results


def evaluate_model_on_dataset(model, dataset_path, cells_max, cell_fill, crop_size, binary_path, device='cuda'):
    """Evaluate a model on a dataset and calculate all metrics.

    Parameters
    ----------
    model : FinetunedSAM
        Model to evaluate
    dataset_path : str
        Path to dataset
    cells_max : float
        Cell peak threshold
    cell_fill : float
        Cell fill threshold
    crop_size : int
        Crop size for sliding window
    binary_path : str
        Path to CTC evaluation binaries
    device : str
        Device to use ('cuda' or 'cpu')

    Returns
    -------
    dict
        Dictionary containing all metrics
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

    from samcell.pipeline import SlidingWindowPipeline
    import uuid
    import torch

    # Load dataset
    img_path = os.path.join(dataset_path, 'imgs.npy')
    ann_path = os.path.join(dataset_path, 'anns.npy')

    imgs_original = np.load(img_path)
    gt_data = np.load(ann_path)

    # Handle different array shapes for images
    if len(imgs_original.shape) == 4:
        if imgs_original.shape[3] > 1:
            imgs = imgs_original[:, :, :, 0].copy()
        else:
            imgs = imgs_original[:, :, :, 0].copy()
    elif len(imgs_original.shape) == 3:
        imgs = imgs_original.copy()
    else:
        raise ValueError(f"Unexpected image array shape: {imgs_original.shape}")

    # Initialize pipeline
    pipeline = SlidingWindowPipeline(
        model=model,
        device=device,
        crop_size=crop_size,
        cells_max=cells_max,
        cell_fill=cell_fill
    )

    # Generate predictions
    predictions = []
    for img in imgs:
        label = pipeline.run(img)
        predictions.append(label)
    pred_data = np.array(predictions)

    # Create evaluation structure
    eval_uuid = uuid.uuid4().hex
    eval_base = f"eval_{os.path.basename(dataset_path)}_{cells_max}_{cell_fill}_{crop_size}_{eval_uuid}"
    gt_seg_dir, gt_tra_dir, res_dir = create_evaluation_structure(eval_base)

    # Calculate StarDist metrics
    try:
        stardist_metrics = calculate_stardist_metrics(gt_data, pred_data)
    except Exception as e:
        print(f"Error calculating stardist metrics: {e}")
        stardist_metrics = {}

    # Save in CTC format
    add_to_cell_tracking_challenge_format(gt_data, pred_data, gt_seg_dir, gt_tra_dir, res_dir)

    # Run CTC evaluation
    sequence = "01"
    num_digits = "04"
    seg_value = run_evaluation("SEGMeasure", eval_base, binary_path, sequence, num_digits)
    det_value = run_evaluation("DETMeasure", eval_base, binary_path, sequence, num_digits)
    csb_value = (seg_value + det_value) / 2 if seg_value is not None and det_value is not None else None

    # Cleanup
    try:
        if os.path.exists(eval_base):
            shutil.rmtree(eval_base)
    except Exception as e:
        print(f"Warning: Could not delete evaluation directory {eval_base}: {e}")

    # Cleanup GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Combine results
    results = {
        "seg_value": seg_value,
        "det_value": det_value,
        "csb_value": csb_value,
    }
    results.update(stardist_metrics)

    return results
