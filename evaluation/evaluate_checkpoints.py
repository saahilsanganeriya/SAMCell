"""
Batch evaluation of SAMCell model checkpoints from WandB.

This script downloads checkpoints from WandB and evaluates them on specified datasets
with various threshold parameter combinations.
"""

import wandb
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import torch
from tqdm import tqdm
import uuid

# Add parent directory to path to import samcell
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from samcell.model import FinetunedSAM
from samcell.pipeline import SlidingWindowPipeline
from eval_utils import (
    run_evaluation,
    create_evaluation_structure,
    add_to_cell_tracking_challenge_format
)

def generate_predictions(model, dataset_path, cells_max, cell_fill, crop_size=256):
    """Generate predictions using the pipeline with specified parameters."""
    img_path = os.path.join(dataset_path, 'imgs.npy')
    imgs_original = np.load(img_path)

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

    # Initialize pipeline with parameters
    pipeline = SlidingWindowPipeline(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        crop_size=crop_size,
        cells_max=cells_max,
        cell_fill=cell_fill
    )

    # Generate predictions
    predictions = []
    for img in imgs:
        label = pipeline.run(img)
        predictions.append(label)

    return np.array(predictions)

def evaluate_model(model, dataset_path, cells_max, cell_fill, binary_path, crop_size=256):
    """Evaluate a model on a dataset with given parameters."""
    # Load dataset
    ann_path = os.path.join(dataset_path, "anns.npy")
    gt_data = np.load(ann_path)

    # Create evaluation structure with unique name
    eval_uuid = uuid.uuid4().hex
    eval_base = f"eval_{os.path.basename(dataset_path)}_{cells_max}_{cell_fill}_{eval_uuid}"
    gt_seg_dir, gt_tra_dir, res_dir = create_evaluation_structure(eval_base)

    # Generate predictions
    pred_data = generate_predictions(model, dataset_path, cells_max, cell_fill, crop_size)

    # Save in CTC format
    add_to_cell_tracking_challenge_format(gt_data, pred_data, gt_seg_dir, gt_tra_dir, res_dir)

    # Run evaluation
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

    # Print results immediately
    print(f"\nResults for {os.path.basename(dataset_path)}:")
    print(f"  cells_max: {cells_max:.2f}, cell_fill: {cell_fill:.2f}")
    print(f"  SEG: {seg_value:.4f}, DET: {det_value:.4f}, CSB: {csb_value:.4f}")

    return {
        "seg_value": seg_value,
        "det_value": det_value,
        "csb_value": csb_value
    }

def main():
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the project
    runs = api.runs("saahilsanganeria666-georgia-institute-of-technology/SAMCell1.0")
    
    # Parameters to evaluate
    cells_max_values = [round(x, 2) for x in np.arange(0.4, 0.76, 0.01)]
    cell_fill_values = [round(x, 2) for x in np.arange(0, 0.11, 0.01)]
    
    # Datasets to evaluate on
    datasets = {
        "PBL_HEK": "datasets/PBL_HEK",
        "PBL_N2A": "datasets/PBL_N2A"
    }
    
    # Binary path
    binary_path = "src/cell-tracking-binaries"
    
    # Create results dataframe
    results = []
    
    # Save results periodically
    def save_results():
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results.csv", index=False)
        print("\nIntermediate results saved to evaluation_results.csv")
    
    try:
        for run in runs:
            print(f"\nProcessing run: {run.name}")
            
            # Get checkpoints
            checkpoints = []
            for artifact in run.logged_artifacts():
                if artifact.type == "model":
                    checkpoints.append(artifact)
            
            # Sort checkpoints by epoch
            checkpoints.sort(key=lambda x: int(x.name.split('_')[-1]) if x.name != "best_model" and x.name != "final_model" else float('inf'))
            
            for checkpoint in checkpoints:
                print(f"\nEvaluating checkpoint: {checkpoint.name}")
                
                # Download and load model
                checkpoint_dir = checkpoint.download()
                model_path = os.path.join(checkpoint_dir, "model.pt")
                
                # Load model
                model = FinetunedSAM('facebook/sam-vit-base')
                model.get_model().load_state_dict(torch.load(model_path))
                model.get_model().eval()
                
                for dataset_name, dataset_path in datasets.items():
                    print(f"\nEvaluating on {dataset_name}")
                    
                    for cells_max in tqdm(cells_max_values, desc="cells_max"):
                        for cell_fill in cell_fill_values:
                            metrics = evaluate_model(
                                model,
                                dataset_path,
                                cells_max,
                                cell_fill,
                                binary_path
                            )
                            
                            # Add to results
                            result = {
                                "run_name": run.name,
                                "checkpoint": checkpoint.name,
                                "dataset": dataset_name,
                                "cells_max": cells_max,
                                "cell_fill": cell_fill,
                                "seg_value": metrics["seg_value"],
                                "det_value": metrics["det_value"],
                                "csb_value": metrics["csb_value"]
                            }
                            results.append(result)
                            
                            # Save results every 100 evaluations
                            if len(results) % 100 == 0:
                                save_results()
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted. Saving current results...")
        save_results()
        raise
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Saving current results...")
        save_results()
        raise
    
    # Final save
    save_results()
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 