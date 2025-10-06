"""
Comprehensive ablation studies for SAMCell.

This script runs systematic ablation studies including:
- Patch size comparisons
- Dataset combination comparisons
- Model architecture comparisons (SAM Base vs Large)
- Pretraining vs random initialization
- Per-epoch performance analysis
"""

import wandb
import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import sys
import datetime
import traceback
import concurrent.futures
from torch.multiprocessing import set_start_method
from threading import Lock
import csv
import argparse
from tqdm import tqdm

# Add paths to import samcell and eval_utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from samcell.model import FinetunedSAM
from samcell.pipeline import SlidingWindowPipeline
from eval_utils import (
    run_evaluation,
    create_evaluation_structure,
    add_to_cell_tracking_challenge_format,
    calculate_stardist_metrics
)

# Try to login to wandb
try:
    wandb.login()
except:
    print("Warning: WandB login failed. Make sure you're logged in.")

# Try to set multiprocessing start method
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set by the parent process

# Create log file with timestamp
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = os.path.join(log_dir, f"ablation_studies_{timestamp}.log")

# Set up logging
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)

print(f"Starting ablation studies extended at {timestamp}")
print(f"Log file: {log_file}")

# Register cleanup on normal exit
import atexit
def cleanup():
    print("Closing log file")
    if hasattr(sys.stdout, "log"):
        sys.stdout.log.close()
    
atexit.register(cleanup)

# Global variables for the ablation studies
CELLS_MAX = 0.46  # Best value from previous studies
CELL_FILL = 0.06  # Best value from previous studies
BINARY_PATH = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCellv2.0.1/src/cell-tracking-binaries"
ABLATION_RESULTS_DIR = "ablation_results"
os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)

# Datasets to evaluate on
DATASETS = {
    "PBL_HEK": "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell/datasets/PBL-HEK"
    #,"PBL_N2A": "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell/datasets/PBL-N2A"
    #,"CellPose-test": "/home/ubuntu/SAMCell/datasets/CellPose-test"
}

# Create file lock for thread-safe CSV writes
csv_file_lock = Lock()

# Utility functions (run_evaluation, create_evaluation_structure, etc.)
# are imported from eval_utils at the top of this file

def generate_predictions(model, dataset_path, cells_max, cell_fill, crop_size, batch_size=5):
    """Generate predictions using the pipeline with specified parameters and GPU batching."""
    img_path = os.path.join(dataset_path, 'imgs.npy')
    
    # Load images and handle different array shapes
    imgs_original = np.load(img_path)
    print(f"Original image array shape: {imgs_original.shape}")
    
    # Handle different array shapes:
    # If 4D (N, H, W, C), take first channel if C > 1
    # If 3D (N, H, W), use as is
    if len(imgs_original.shape) == 4:
        print(f"4D image array detected with {imgs_original.shape[3]} channels")
        if imgs_original.shape[3] > 1:
            print(f"Using first channel from 4D array")
            imgs = imgs_original[:, :, :, 0].copy()  # Make a copy instead of a view
        else:
            print(f"Using single channel from 4D array")
            imgs = imgs_original[:, :, :, 0].copy()  # Make a copy instead of a view
    elif len(imgs_original.shape) == 3:
        print(f"3D image array detected, using as-is")
        imgs = imgs_original.copy()  # Make a copy instead of a view
    else:
        raise ValueError(f"Unexpected image array shape: {imgs_original.shape}")
    
    # Force release of original array
    del imgs_original
    import gc
    gc.collect()
    
    print(f"Processed image array shape: {imgs.shape}")
    
    # Initialize pipeline with parameters
    pipeline = SlidingWindowPipeline(
        model=model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        crop_size=crop_size,
        cells_max=cells_max,
        cell_fill=cell_fill
    )
    
    # Generate predictions for all images
    all_predictions = np.zeros(imgs.shape, dtype=np.int32)
    
    # Process images in batches to better utilize GPU resources
    total_images = len(imgs)
    num_batches = (total_images + batch_size - 1) // batch_size
    
    # Create a single progress bar for the entire dataset
    with tqdm(total=total_images, desc=f"Generating predictions for {os.path.basename(dataset_path)}", unit="img") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_images)
            batch_imgs = imgs[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx
            
            print(f"\nBatch {batch_idx+1}/{num_batches} (images {start_idx}-{end_idx-1})")
            
            # Batch processing of images
            for img_idx, img in enumerate(batch_imgs):
                global_img_idx = start_idx + img_idx
                
                # Get prediction for this image
                prediction = pipeline.run(img)
                all_predictions[global_img_idx] = prediction
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({"Image": global_img_idx, "Batch": f"{batch_idx+1}/{num_batches}"})
            
            # Clear CUDA cache after each batch to prevent memory fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_predictions

def evaluate_checkpoint(run_id, checkpoint_name, dataset_name, dataset_path, crop_size, cells_max, cell_fill, binary_path, csv_filename, model_type):
    """Evaluate a specific checkpoint on a dataset with given parameters."""
    print(f"\nEvaluating checkpoint {checkpoint_name} on {dataset_name} with crop_size={crop_size}")
    
    try:
        # Initialize wandb API
        api = wandb.Api()
        
        # Get the run and model artifact
        run = api.run(f"saahilsanganeria666-georgia-institute-of-technology/SAMCell1.0/{run_id}")
        
        # Fix: Get the artifact properly - first get the artifact with api.artifact, then use it
        # Try fetching :v0 first, assuming it's the standard for epoch checkpoints, then fallback.
        checkpoint_dir = None
        model_path = None
        artifact_found = False
        tags_to_try = ["v0", "latest"] # Prioritize v0

        for tag in tags_to_try:
            try:
                artifact_path = f"{run.entity}/{run.project}/{checkpoint_name}:{tag}"
                print(f"Attempting to download artifact: {artifact_path}")
                artifact = api.artifact(artifact_path)
                download_dir = artifact.download()
                potential_model_path = os.path.join(download_dir, "model.pt")
                
                if os.path.exists(potential_model_path):
                    print(f"Successfully downloaded artifact and found model.pt using tag ':{tag}'")
                    checkpoint_dir = download_dir
                    model_path = potential_model_path
                    artifact_found = True
                    break # Stop searching once found
                else:
                    print(f"Warning: model.pt not found in downloaded artifact {artifact_path}. Trying next tag.")
                    # Clean up the directory if model.pt wasn't found to avoid clutter
                    try:
                        if os.path.exists(download_dir):
                            shutil.rmtree(download_dir)
                    except Exception as cleanup_error:
                         print(f"Warning: Could not cleanup incomplete download directory {download_dir}: {cleanup_error}")

            except wandb.errors.CommError as e:
                print(f"Wandb communication error fetching artifact {artifact_path}: {e}. Trying next tag.")
            except Exception as e:
                print(f"Unexpected error fetching artifact {artifact_path}: {e}. Trying next tag.")
                traceback.print_exc() # Print traceback for unexpected errors

        if not artifact_found:
            raise FileNotFoundError(f"Could not find or download a valid artifact for '{checkpoint_name}' with tags {tags_to_try} containing model.pt")
        
        # model_path is now set if found
        # checkpoint_dir is the directory containing model.pt
        
        # Load model with correct architecture
        model = FinetunedSAM(model_type)
        model.load_weights(model_path)
        model.get_model().eval()
        if torch.cuda.is_available():
            model.get_model().cuda()
        
        # Load dataset
        ann_path = os.path.join(dataset_path, "anns.npy")
        gt_data = np.load(ann_path)
        
        # Check if we're in Study 5 (ablation_study == "per_epoch") and handle cached predictions
        is_study_5 = "epoch" in checkpoint_name
        if is_study_5:
            # Create directory for cached segmentations
            seg_cache_dir = os.path.join(ABLATION_RESULTS_DIR, "segmentations")
            os.makedirs(seg_cache_dir, exist_ok=True)
            
            # Define a unique filename for this dataset + checkpoint combination
            safe_checkpoint_name = checkpoint_name.replace("/", "_").replace(":", "_")
            safe_dataset_name = os.path.basename(dataset_path)
            seg_cache_file = os.path.join(
                seg_cache_dir, 
                f"{run_id}_{safe_checkpoint_name}_{safe_dataset_name}_{crop_size}.npy"
            )
            
            # Check if cached segmentation exists
            if os.path.exists(seg_cache_file):
                print(f"Loading cached segmentation from {seg_cache_file}")
                try:
                    pred_data = np.load(seg_cache_file)
                    print(f"Successfully loaded cached segmentation of shape {pred_data.shape}")
                except Exception as e:
                    print(f"Error loading cached segmentation: {e}, regenerating predictions")
                    pred_data = generate_predictions(model, dataset_path, cells_max, cell_fill, crop_size)
                    # Save the predictions for future use
                    np.save(seg_cache_file, pred_data)
                    print(f"Saved segmentation to {seg_cache_file}")
            else:
                print(f"No cached segmentation found, generating predictions")
                pred_data = generate_predictions(model, dataset_path, cells_max, cell_fill, crop_size)
                # Save the predictions for future use
                np.save(seg_cache_file, pred_data)
                print(f"Saved segmentation to {seg_cache_file}")
        else:
            # For other studies, generate predictions without caching
            pred_data = generate_predictions(model, dataset_path, cells_max, cell_fill, crop_size)
        
        # Create evaluation structure
        eval_uuid = uuid.uuid4().hex
        eval_base = f"eval_{os.path.basename(dataset_path)}_{cells_max}_{cell_fill}_{crop_size}_{eval_uuid}"
        gt_seg_dir, gt_tra_dir, res_dir = create_evaluation_structure(eval_base)
        
        # Calculate stardist metrics
        try:
            stardist_metrics = calculate_stardist_metrics(gt_data, pred_data)
        except Exception as e:
            print(f"Error calculating stardist metrics: {e}")
            stardist_metrics = {}
        
        # Save in CTC format
        add_to_cell_tracking_challenge_format(gt_data, pred_data, gt_seg_dir, gt_tra_dir, res_dir)
        
        # Run evaluation
        sequence = "01"
        num_digits = "04"
        seg_value = run_evaluation("SEGMeasure", eval_base, binary_path, sequence, num_digits)
        det_value = run_evaluation("DETMeasure", eval_base, binary_path, sequence, num_digits)
        csb_value = (seg_value + det_value) / 2 if seg_value is not None and det_value is not None else None
        
        # Cleanup - safely delete directories
        try:
            if os.path.exists(eval_base):
                shutil.rmtree(eval_base)
        except Exception as e:
            print(f"Warning: Could not delete evaluation directory {eval_base}: {e}")
        
        try:
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            else:
                print(f"Warning: Checkpoint directory not found at {checkpoint_dir}")
                # Try to find and clean up any wandb artifact directories
                artifact_dirname = checkpoint_name.replace(':', '_')
                for root, dirs, _ in os.walk('artifacts'):
                    for d in dirs:
                        if checkpoint_name in d or artifact_dirname in d:
                            try:
                                dir_path = os.path.join(root, d)
                                print(f"Cleaning up found artifact directory: {dir_path}")
                                shutil.rmtree(dir_path)
                            except Exception as cleanup_error:
                                print(f"Could not clean up directory {d}: {cleanup_error}")
        except Exception as e:
            print(f"Warning: Could not delete checkpoint directory {checkpoint_dir}: {e}")
        
        # Print results immediately
        print(f"\nResults for {dataset_name}:")
        print(f"  Run: {run.name}, Checkpoint: {checkpoint_name}")
        print(f"  cells_max: {cells_max:.2f}, cell_fill: {cell_fill:.2f}, crop_size: {crop_size}")
        print(f"  SEG: {seg_value:.4f}, DET: {det_value:.4f}, CSB: {csb_value:.4f}")
        
        if stardist_metrics:
            print(f"  StarDist IoU@0.5: precision={stardist_metrics['precision']:.4f}, "
                  f"recall={stardist_metrics['recall']:.4f}, f1={stardist_metrics['f1']:.4f}")
        
        # Create result dictionary
        result = {
            "run_name": run.name,
            "run_id": run_id,
            "checkpoint_name": checkpoint_name,
            "model_type": model_type,
            "dataset": dataset_name,
            "crop_size": crop_size,
            "cells_max": cells_max,
            "cell_fill": cell_fill,
            "seg_value": seg_value,
            "det_value": det_value,
            "csb_value": csb_value,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ablation_study": "crop_size" if model_type == "facebook/sam-vit-base" else "model_type"
        }
        
        # Add stardist metrics
        for tau_key, metrics in stardist_metrics.items():
            if isinstance(metrics, dict):
                for metric_name, metric_value in metrics.items():
                    result[f"stardist_{tau_key}_{metric_name}"] = metric_value
            else:
                result[f"stardist_{tau_key}"] = metrics
        
        # Save result to CSV
        save_result_to_csv(result, csv_filename)
        
        # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
        return True, result
        
    except Exception as e:
        print(f"Error evaluating checkpoint {checkpoint_name} on {dataset_name}: {str(e)}")
        traceback.print_exc()
        return False, None

def save_result_to_csv(result, csv_filename):
    """Save a single result to CSV in a thread-safe manner."""
    with csv_file_lock:
        file_exists = os.path.exists(csv_filename)
        try:
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=result.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)
            print(f"  Result saved to CSV: {csv_filename}")
        except Exception as e:
            print(f"Error saving result to CSV: {e}")

def run_study_1_patch_sizes():
    """Study 1: Evaluate different patch sizes (128, 256, 512)."""
    print("\n" + "="*80)
    print("STUDY 1: DIFFERENT PATCH SIZES")
    print("="*80)
    
    # Parameters for this study
    run_id = "xeaxlskz"  # CellPose-train+LiveCell-train-sam-vit-base-bs8-lr1.0e-04
    checkpoint_name = "best-model-run-xeaxlskz"
    model_type = "facebook/sam-vit-base"
    crop_sizes = [128, 256, 512]
    
    # CSV file for results
    csv_filename = os.path.join(ABLATION_RESULTS_DIR, "study_1_patch_sizes.csv")
    
    # Create tasks for parallel processing
    tasks = []
    for dataset_name, dataset_path in DATASETS.items():
        for crop_size in crop_sizes:
            tasks.append((
                run_id, checkpoint_name, dataset_name, dataset_path, 
                crop_size, CELLS_MAX, CELL_FILL, BINARY_PATH, 
                csv_filename, model_type
            ))
    
    # Track which tasks have been completed
    completed_combinations = set()
    
    # First, check if we have existing results that we can use
    if os.path.exists(csv_filename):
        try:
            existing_df = pd.read_csv(csv_filename)
            for _, row in existing_df.iterrows():
                combo = (row['dataset'], row['crop_size'])
                completed_combinations.add(combo)
            print(f"Found {len(completed_combinations)} completed combinations in existing results.")
        except Exception as e:
            print(f"Error reading existing results: {e}")
    
    # Filter out tasks that have already been completed
    filtered_tasks = []
    for task in tasks:
        dataset_name = task[2]
        crop_size = task[4]
        if (dataset_name, crop_size) not in completed_combinations:
            filtered_tasks.append(task)
    
    if not filtered_tasks:
        print("All combinations have already been evaluated!")
    else:
        print(f"Need to evaluate {len(filtered_tasks)} combinations.")
        
    # Track tasks that failed and need retrying
    failed_tasks = []
    max_retries = 2  # Maximum number of retries for failed tasks
    
    # Use ThreadPoolExecutor for parallel processing with reduced parallelism
    completed_tasks = 0
    total_tasks = len(filtered_tasks)
    
    if total_tasks > 0:
        print(f"Processing {total_tasks} tasks with up to 2 workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(evaluate_checkpoint, *task) for task in filtered_tasks]
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    success, result = future.result()
                    task_idx = futures.index(future)
                    task = filtered_tasks[task_idx]
                    dataset_name = task[2]
                    crop_size = task[4]
                    
                    if success:
                        completed_tasks += 1
                        completed_combinations.add((dataset_name, crop_size))
                        print(f"Completed {completed_tasks}/{total_tasks} tasks for Study 1")
                        print(f"  Success: {dataset_name} with crop_size={crop_size}")
                    else:
                        print(f"Task failed: {dataset_name} with crop_size={crop_size}")
                        failed_tasks.append(task)
                except Exception as e:
                    task_idx = futures.index(future)
                    task = filtered_tasks[task_idx]
                    dataset_name = task[2]
                    crop_size = task[4]
                    print(f"Error in task for {dataset_name} with crop_size={crop_size}: {e}")
                    failed_tasks.append(task)
    
    # Retry failed tasks
    retry_count = 0
    while failed_tasks and retry_count < max_retries:
        retry_count += 1
        print(f"\nRetrying {len(failed_tasks)} failed tasks (attempt {retry_count}/{max_retries})...")
        
        retry_failed = []
        for task in failed_tasks:
            dataset_name = task[2]
            crop_size = task[4]
            print(f"Retrying: {dataset_name} with crop_size={crop_size}")
            
            try:
                success, result = evaluate_checkpoint(*task)
                if success:
                    completed_combinations.add((dataset_name, crop_size))
                    print(f"  Success on retry: {dataset_name} with crop_size={crop_size}")
                else:
                    print(f"  Failed again: {dataset_name} with crop_size={crop_size}")
                    retry_failed.append(task)
            except Exception as e:
                print(f"  Error on retry for {dataset_name} with crop_size={crop_size}: {e}")
                retry_failed.append(task)
        
        failed_tasks = retry_failed
    
    # Check if all combinations were evaluated
    all_combinations = [(dataset_name, crop_size) 
                        for dataset_name in DATASETS.keys() 
                        for crop_size in crop_sizes]
    
    missing_combinations = set(all_combinations) - completed_combinations
    if missing_combinations:
        print(f"\nWARNING: {len(missing_combinations)} combinations could not be evaluated after retries:")
        for combo in missing_combinations:
            print(f"  Missing: {combo[0]} with crop_size={combo[1]}")
    
    # Analyze results to determine best patch size
    try:
        df = pd.read_csv(csv_filename)
        
        # Check for missing values
        for dataset_name in DATASETS.keys():
            for crop_size in crop_sizes:
                subset = df[(df['dataset'] == dataset_name) & (df['crop_size'] == crop_size)]
                if len(subset) == 0:
                    print(f"WARNING: No results for {dataset_name} with crop_size={crop_size}")
        
        # Group by crop_size and calculate average CSB value across datasets
        avg_by_patch = df.groupby('crop_size')['csb_value'].mean().reset_index()
        if len(avg_by_patch) < len(crop_sizes):
            print("WARNING: Not all crop sizes have results. Using partial data for analysis.")
            
        best_patch_size = int(avg_by_patch.loc[avg_by_patch['csb_value'].idxmax(), 'crop_size'])
        print(f"\nStudy 1 Results: Best patch size is {best_patch_size}")
        print(f"Average CSB values by patch size:")
        print(avg_by_patch)
        
        # If we're missing results for the best patch size for any dataset, warn about it
        for dataset_name in DATASETS.keys():
            best_size_results = df[(df['dataset'] == dataset_name) & (df['crop_size'] == best_patch_size)]
            if len(best_size_results) == 0:
                print(f"WARNING: No results for {dataset_name} with best patch size {best_patch_size}")
                
        return best_patch_size
    except Exception as e:
        print(f"Error analyzing Study 1 results: {e}")
        print("Using default patch size of 256")
        return 256  # Default to 256 if analysis fails

def run_study_2_dataset_combos(best_patch_size):
    """Study 2: Compare different dataset combinations for training."""
    print("\n" + "="*80)
    print(f"STUDY 2: DATASET COMBINATIONS (using patch size {best_patch_size})")
    print("="*80)
    
    # Parameters for this study
    run_configs = [
        # CellPose+LiveCell
        {"run_id": "xeaxlskz", "checkpoint_name": "best-model-run-xeaxlskz", "model_type": "facebook/sam-vit-base"},
        # LiveCell only
        {"run_id": "3jrd913f", "checkpoint_name": "best-model-run-3jrd913f", "model_type": "facebook/sam-vit-base"},
        # CellPose only
        {"run_id": "wgplg3x2", "checkpoint_name": "best-model-run-wgplg3x2", "model_type": "facebook/sam-vit-base"}
    ]
    
    # CSV file for results
    csv_filename = os.path.join(ABLATION_RESULTS_DIR, "study_2_dataset_combos.csv")
    
    # Create tasks for parallel processing
    tasks = []
    for config in run_configs:
        for dataset_name, dataset_path in DATASETS.items():
            tasks.append((
                config["run_id"], config["checkpoint_name"], dataset_name, dataset_path, 
                best_patch_size, CELLS_MAX, CELL_FILL, BINARY_PATH, 
                csv_filename, config["model_type"]
            ))
    
    # Use ThreadPoolExecutor for parallel processing
    completed_tasks = 0
    total_tasks = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(evaluate_checkpoint, *task) for task in tasks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                success, result = future.result()
                completed_tasks += 1
                print(f"Completed {completed_tasks}/{total_tasks} tasks for Study 2")
            except Exception as e:
                print(f"Error in task for Study 2: {e}")
    
    # Analyze results
    try:
        df = pd.read_csv(csv_filename)
        # Group by run_name and calculate average CSB value across datasets
        avg_by_run = df.groupby('run_name')['csb_value'].mean().reset_index()
        best_run = avg_by_run.loc[avg_by_run['csb_value'].idxmax(), 'run_name']
        print(f"\nStudy 2 Results: Best dataset combination is from {best_run}")
        print(f"Average CSB values by dataset combination:")
        print(avg_by_run)
    except Exception as e:
        print(f"Error analyzing Study 2 results: {e}")

def run_study_3_sam_models(best_patch_size):
    """Study 3: Compare SAM Base vs SAM Large models."""
    print("\n" + "="*80)
    print(f"STUDY 3: SAM BASE VS SAM LARGE (using patch size {best_patch_size})")
    print("="*80)
    
    # Parameters for this study
    run_configs = [
        # SAM Base
        {"run_id": "xeaxlskz", "checkpoint_name": "best-model-run-xeaxlskz", "model_type": "facebook/sam-vit-base"},
        # SAM Large
        {"run_id": "5tdyjcjc", "checkpoint_name": "best-model-run-5tdyjcjc", "model_type": "facebook/sam-vit-large"}
    ]
    
    # CSV file for results
    csv_filename = os.path.join(ABLATION_RESULTS_DIR, "study_3_sam_models.csv")
    
    # Create tasks for parallel processing
    tasks = []
    for config in run_configs:
        for dataset_name, dataset_path in DATASETS.items():
            tasks.append((
                config["run_id"], config["checkpoint_name"], dataset_name, dataset_path, 
                best_patch_size, CELLS_MAX, CELL_FILL, BINARY_PATH, 
                csv_filename, config["model_type"]
            ))
    
    # Use ThreadPoolExecutor for parallel processing
    completed_tasks = 0
    total_tasks = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(evaluate_checkpoint, *task) for task in tasks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                success, result = future.result()
                completed_tasks += 1
                print(f"Completed {completed_tasks}/{total_tasks} tasks for Study 3")
            except Exception as e:
                print(f"Error in task for Study 3: {e}")
    
    # Analyze results
    try:
        df = pd.read_csv(csv_filename)
        # Group by model_type and calculate average CSB value across datasets
        avg_by_model = df.groupby('model_type')['csb_value'].mean().reset_index()
        best_model = avg_by_model.loc[avg_by_model['csb_value'].idxmax(), 'model_type']
        print(f"\nStudy 3 Results: Best model is {best_model}")
        print(f"Average CSB values by model type:")
        print(avg_by_model)
    except Exception as e:
        print(f"Error analyzing Study 3 results: {e}")

def run_study_4_pretraining(best_patch_size):
    """Study 4: Compare pretraining vs random initialization."""
    print("\n" + "="*80)
    print(f"STUDY 4: PRETRAINING VS RANDOM INITIALIZATION (using patch size {best_patch_size})")
    print("="*80)
    
    # Parameters for this study
    run_configs = [
        # Pretrained weights
        {"run_id": "xeaxlskz", "checkpoint_name": "best-model-run-xeaxlskz", "model_type": "facebook/sam-vit-base"},
        # Random weights
        {"run_id": "lpaw1rdp", "checkpoint_name": "best-model-run-lpaw1rdp", "model_type": "facebook/sam-vit-base"}
    ]
    
    # CSV file for results
    csv_filename = os.path.join(ABLATION_RESULTS_DIR, "study_4_pretraining.csv")
    
    # Create tasks for parallel processing
    tasks = []
    for config in run_configs:
        for dataset_name, dataset_path in DATASETS.items():
            tasks.append((
                config["run_id"], config["checkpoint_name"], dataset_name, dataset_path, 
                best_patch_size, CELLS_MAX, CELL_FILL, BINARY_PATH, 
                csv_filename, config["model_type"]
            ))
    
    # Use ThreadPoolExecutor for parallel processing
    completed_tasks = 0
    total_tasks = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(evaluate_checkpoint, *task) for task in tasks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                success, result = future.result()
                completed_tasks += 1
                print(f"Completed {completed_tasks}/{total_tasks} tasks for Study 4")
            except Exception as e:
                print(f"Error in task for Study 4: {e}")
    
    # Analyze results
    try:
        df = pd.read_csv(csv_filename)
        # Group by run_name (which indicates pretrained vs random) and calculate average CSB value
        avg_by_init = df.groupby('run_name')['csb_value'].mean().reset_index()
        best_init = avg_by_init.loc[avg_by_init['csb_value'].idxmax(), 'run_name']
        print(f"\nStudy 4 Results: Best initialization is from {best_init}")
        print(f"Average CSB values by initialization:")
        print(avg_by_init)
    except Exception as e:
        print(f"Error analyzing Study 4 results: {e}")

def evaluate_epoch_checkpoint(run_id, epoch, dataset_name, dataset_path, crop_size, cells_max, cell_fill, binary_path, csv_filename, model_type):
    """Evaluate a specific epoch checkpoint on a dataset."""
    checkpoint_name = f"model-checkpoint-epoch-{epoch}-run-{run_id}"
    print(f"\nEvaluating epoch {epoch} checkpoint on {dataset_name}")
    
    result = evaluate_checkpoint(
        run_id, checkpoint_name, dataset_name, dataset_path, 
        crop_size, cells_max, cell_fill, binary_path, csv_filename, model_type
    )
    
    # Add epoch information to the result if successful
    if result[0] and result[1]:
        result[1]["epoch"] = epoch
        result[1]["ablation_study"] = "per_epoch"
        
        # Re-save with the additional information
        save_result_to_csv(result[1], csv_filename)
    
    return result

def run_study_5_per_epoch(best_patch_size):
    """Study 5: Evaluate performance changes per epoch."""
    print("\n" + "="*80)
    print(f"STUDY 5: PER-EPOCH CHANGES (using patch size {best_patch_size})")
    print("="*80)
    
    # Parameters for this study
    run_configs = [
        # Pretrained - epochs up to 34 (from the CSV data)
        {"run_id": "xeaxlskz", "epochs": [20], "model_type": "facebook/sam-vit-base"}, # list(range(5, 35, 5))
        # Random weights - epochs up to 39 (from the CSV data)
        #{"run_id": "lpaw1rdp", "epochs": [0], "model_type": "facebook/sam-vit-base"} # list(range(5, 40, 5))
    ]
    
    # CSV file for results
    csv_filename = os.path.join(ABLATION_RESULTS_DIR, "study_5_per_epoch_epochs_0.csv")
    
    # Use available datasets instead of hardcoding
    available_datasets = list(DATASETS.items())
    if not available_datasets:
        print("Error: No datasets available for Study 5")
        return
    
    # Create tasks for parallel processing
    tasks = []
    for config in run_configs:
        for epoch in config["epochs"]:
            for dataset_name, dataset_path in available_datasets:
                tasks.append((
                    config["run_id"], epoch, dataset_name, dataset_path, 
                    best_patch_size, CELLS_MAX, CELL_FILL, BINARY_PATH, 
                    csv_filename, config["model_type"]
                ))
    
    print(f"Running Study 5 with {len(available_datasets)} datasets: {list(DATASETS.keys())}")
    print(f"Total tasks: {len(tasks)} (epochs × runs × datasets)")
    
    # Use ThreadPoolExecutor for parallel processing
    completed_tasks = 0
    total_tasks = len(tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(evaluate_epoch_checkpoint, *task) for task in tasks]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                success, result = future.result()
                completed_tasks += 1
                print(f"Completed {completed_tasks}/{total_tasks} tasks for Study 5")
            except Exception as e:
                print(f"Error in task for Study 5: {e}")
    
    # Analyze results
    try:
        df = pd.read_csv(csv_filename)
        # Plot performance by epoch for each run
        grouped = df.groupby(['run_name', 'epoch'])['csb_value'].mean().reset_index()
        
        # Save results to CSV for plotting outside the script
        epoch_summary_file = os.path.join(ABLATION_RESULTS_DIR, "study_5_epoch_summary.csv")
        grouped.to_csv(epoch_summary_file, index=False)
        
        print(f"\nStudy 5 Results: Performance by epoch saved to {epoch_summary_file}")
        print("Performance by epoch and run:")
        for run, group in grouped.groupby('run_name'):
            print(f"\n{run}:")
            for _, row in group.iterrows():
                print(f"  Epoch {row['epoch']}: CSB = {row['csb_value']:.4f}")
    except Exception as e:
        print(f"Error analyzing Study 5 results: {e}")

def main():
    print("Starting ablation studies extended")
    print(f"Fixed parameters: cells_max={CELLS_MAX}, cell_fill={CELL_FILL}")
    
    # Add global declaration at the beginning of the function
    global DATASETS
    
    # Add argparse to allow running specific studies only
    parser = argparse.ArgumentParser(description='Run ablation studies for SAMCell')
    parser.add_argument('--study', type=int, choices=[1, 2, 3, 4, 5], 
                      help='Run only a specific study (1-5)')
    parser.add_argument('--patch-size', type=int, default=None,
                      help='Override the best patch size from Study 1')
    parser.add_argument('--datasets', type=str, default=None,
                      help='Comma-separated list of datasets to use (e.g., "PBL_HEK,PBL_N2A")')
    args = parser.parse_args()
    
    # Filter datasets if specified
    selected_datasets = DATASETS
    if args.datasets:
        dataset_names = [name.strip() for name in args.datasets.split(',')]
        selected_datasets = {name: path for name, path in DATASETS.items() if name in dataset_names}
        if not selected_datasets:
            print(f"Warning: No valid datasets found in '{args.datasets}'. Using all datasets.")
            selected_datasets = DATASETS
        else:
            print(f"Using selected datasets: {list(selected_datasets.keys())}")
    
    # If a specific study is requested
    if args.study:
        print(f"Running only Study {args.study}")
        
        if args.study == 1:
            # Always run Study 1 if requested
            best_patch_size = run_study_1_patch_sizes()
        else:
            # For studies 2-5, we need a patch size
            best_patch_size = args.patch_size
            if best_patch_size is None:
                # Try to load from Study 1 results
                try:
                    study1_file = os.path.join(ABLATION_RESULTS_DIR, "study_1_patch_sizes.csv")
                    if os.path.exists(study1_file):
                        df = pd.read_csv(study1_file)
                        avg_by_patch = df.groupby('crop_size')['csb_value'].mean().reset_index()
                        best_patch_size = int(avg_by_patch.loc[avg_by_patch['csb_value'].idxmax(), 'crop_size'])
                        print(f"Loaded best patch size {best_patch_size} from Study 1 results")
                    else:
                        print("Study 1 results not found, using default patch size 256")
                        best_patch_size = 256
                except Exception as e:
                    print(f"Error loading Study 1 results: {e}")
                    print("Using default patch size 256")
                    best_patch_size = 256
            
            # Run the specific study with the selected datasets
            original_datasets = DATASETS
            DATASETS = selected_datasets
            
            if args.study == 2:
                run_study_2_dataset_combos(best_patch_size)
            elif args.study == 3:
                run_study_3_sam_models(best_patch_size)
            elif args.study == 4:
                run_study_4_pretraining(best_patch_size)
            elif args.study == 5:
                run_study_5_per_epoch(best_patch_size)
            
            # Restore original datasets
            DATASETS = original_datasets
    else:
        # Run all studies in sequence
        best_patch_size = run_study_1_patch_sizes()
        
        print(f"\nCompleted Study 1. Using best patch size: {best_patch_size} for remaining studies.")
        time.sleep(5)  # Short pause to ensure all resources are freed
        
        run_study_2_dataset_combos(best_patch_size)
        print("\nCompleted Study 2.")
        time.sleep(5)
        
        run_study_3_sam_models(best_patch_size)
        print("\nCompleted Study 3.")
        time.sleep(5)
        
        run_study_4_pretraining(best_patch_size)
        print("\nCompleted Study 4.")
        time.sleep(5)
        
        # For Study 5, use only specified datasets if provided
        if args.datasets:
            print(f"\nRunning Study 5 with selected datasets: {list(selected_datasets.keys())}")
            original_datasets = DATASETS
            DATASETS = selected_datasets
            run_study_5_per_epoch(best_patch_size)
            DATASETS = original_datasets
        else:
            run_study_5_per_epoch(best_patch_size)
    
    print("\nAll ablation studies completed!")
    print(f"Results are saved in the {ABLATION_RESULTS_DIR} directory")

if __name__ == "__main__":
    main() 