#!/usr/bin/env python3
"""
Process Ratcliff Lab Mitotic Spindle Data for SAMCell Training
===============================================================

Converts the exported green channel images and Cellpose masks into
the format required by SAMCell training (imgs.npy and dist_maps.npy).

This script:
1. Loads green channel images from exported_images_ch1/
2. Loads segmentation masks from exported_masks_raw/
3. Resizes to 512x512 (maintains aspect ratio + padding)
4. Converts masks to distance maps
5. Saves as imgs.npy and dist_maps.npy

Usage:
    python process_ratcliff_lab_data.py --input-folder /path/to/images_to_analyze
    python process_ratcliff_lab_data.py --input-folder /path/to/images_for_cellpose_model
    python process_ratcliff_lab_data.py --input-folder /path/to/images_to_analyze --output-dir ../datasets/RatcliffLab
"""

import numpy as np
import cv2
import os
import argparse
from pathlib import Path

# Try to import scipy, use opencv as fallback
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using OpenCV for distance transform")

# Try to import tqdm, use fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def create_distance_map(mask):
    """
    Create distance map from segmentation mask.
    
    Distance map shows the distance from each pixel to the nearest cell boundary.
    Interior of cells has positive distance, background has 0.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask where each cell has a unique label
        
    Returns
    -------
    numpy.ndarray
        Distance map (same shape as mask)
    """
    # Create binary mask (cells vs background)
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Compute distance transform (distance from each pixel to nearest background)
    if SCIPY_AVAILABLE:
        dist_map = distance_transform_edt(binary_mask)
    else:
        # Use OpenCV distance transform as fallback
        dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5).astype(np.float32)
    
    # Normalize to 0-1 range
    if dist_map.max() > 0:
        dist_map = dist_map / dist_map.max()
    
    return dist_map.astype(np.float32)


def create_weight_map(mask, w0=10, sigma=5):
    """
    Create weight map for weighted loss (optional, like U-Net paper).
    
    Emphasizes cell boundaries and separation of touching cells.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask where each cell has a unique label
    w0 : float
        Weight of boundary map
    sigma : float
        Standard deviation for Gaussian
        
    Returns
    -------
    numpy.ndarray
        Weight map (same shape as mask)
    """
    # Simple edge-based weight map
    # Find boundaries using gradient
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1)
    boundaries = (dilated - eroded) > 0
    
    # Create base weight map (1 everywhere)
    weight_map = np.ones_like(mask, dtype=np.float32)
    
    # Increase weight at boundaries
    weight_map[boundaries] = w0
    
    # Apply Gaussian smoothing with OpenCV
    weight_map = cv2.GaussianBlur(weight_map, (0, 0), sigma)
    
    return weight_map


def resize_with_padding(img, target_size=512):
    """
    Resize image to target size while maintaining aspect ratio.
    Pads with zeros to make it square.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W) or (H, W, C)
    target_size : int
        Target size (default: 512)
        
    Returns
    -------
    numpy.ndarray
        Resized and padded image (target_size, target_size)
    """
    h, w = img.shape[:2]
    
    # Calculate scaling factor
    if h > w:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)
    else:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)
    
    # Resize
    if len(img.shape) == 3:
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    
    if len(img.shape) == 3:
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_CONSTANT, value=0)
    else:
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, 
                                       cv2.BORDER_CONSTANT, value=0)
    
    return img_padded


def process_dataset(input_folder, output_dir, target_size=512, create_weights=False):
    """
    Process a dataset folder to create SAMCell training data.
    
    Parameters
    ----------
    input_folder : str or Path
        Folder containing exported_images_ch1/ and exported_masks_raw/
    output_dir : str or Path
        Output directory for imgs.npy and dist_maps.npy
    target_size : int
        Target image size (default: 512)
    create_weights : bool
        Whether to create weight maps (default: False)
    """
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image and mask directories
    img_dir = input_folder / 'exported_images_ch1'
    mask_dir = input_folder / 'exported_masks_raw'
    
    if not img_dir.exists():
        raise ValueError(f"Image directory not found: {img_dir}")
    if not mask_dir.exists():
        raise ValueError(f"Mask directory not found: {mask_dir}")
    
    # Find all images
    img_files = sorted(list(img_dir.glob('*.png')))
    
    if len(img_files) == 0:
        raise ValueError(f"No PNG files found in {img_dir}")
    
    print(f"\nProcessing dataset: {input_folder.name}")
    print(f"Found {len(img_files)} images")
    
    imgs = []
    dist_maps = []
    weight_maps = [] if create_weights else None
    
    for img_file in tqdm(img_files, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_file), cv2.IMREAD_UNCHANGED)
        
        # Find corresponding mask
        img_stem = img_file.stem
        mask_file = mask_dir / f"{img_stem}_mask.png"
        
        if not mask_file.exists():
            print(f"Warning: No mask found for {img_file.name}, skipping")
            continue
        
        # Load mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        
        # Verify dimensions match
        if img.shape[:2] != mask.shape[:2]:
            print(f"Warning: Dimension mismatch for {img_file.name}, skipping")
            continue
        
        # Convert image to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize image and mask
        img_resized = resize_with_padding(img, target_size)
        mask_resized = resize_with_padding(mask, target_size)
        
        # Create distance map from mask
        dist_map = create_distance_map(mask_resized)
        
        # Normalize image to 0-1 range
        img_normalized = img_resized.astype(np.float32)
        if img_normalized.max() > 1.0:
            img_normalized = img_normalized / img_normalized.max()
        
        imgs.append(img_normalized)
        dist_maps.append(dist_map)
        
        # Optionally create weight map
        if create_weights:
            weight_map = create_weight_map(mask_resized)
            weight_maps.append(weight_map)
    
    # Convert to numpy arrays
    imgs = np.array(imgs, dtype=np.float32)
    dist_maps = np.array(dist_maps, dtype=np.float32)
    
    print(f"\nFinal dataset shapes:")
    print(f"  Images: {imgs.shape}")
    print(f"  Distance maps: {dist_maps.shape}")
    
    # Save
    imgs_path = output_dir / 'imgs.npy'
    dist_maps_path = output_dir / 'dist_maps.npy'
    
    np.save(imgs_path, imgs)
    np.save(dist_maps_path, dist_maps)
    
    print(f"\n✓ Saved to:")
    print(f"  {imgs_path}")
    print(f"  {dist_maps_path}")
    
    if create_weights:
        weight_maps = np.array(weight_maps, dtype=np.float32)
        wms_path = output_dir / 'wms.npy'
        np.save(wms_path, weight_maps)
        print(f"  {wms_path}")
        print(f"  Weight maps: {weight_maps.shape}")
    
    # Show some statistics
    print(f"\nDataset statistics:")
    print(f"  Number of samples: {len(imgs)}")
    print(f"  Image size: {target_size}x{target_size}")
    print(f"  Image range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"  Distance map range: [{dist_maps.min():.3f}, {dist_maps.max():.3f}]")
    
    return imgs, dist_maps


def visualize_sample(imgs, dist_maps, idx=0):
    """Visualize a sample from the dataset."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(imgs[idx], cmap='gray')
    axes[0].set_title('Image (Green Channel)')
    axes[0].axis('off')
    
    axes[1].imshow(dist_maps[idx], cmap='jet')
    axes[1].set_title('Distance Map')
    axes[1].axis('off')
    
    # Show overlay
    axes[2].imshow(imgs[idx], cmap='gray')
    axes[2].imshow(dist_maps[idx], cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Convert Ratcliff Lab data to SAMCell training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Process images_to_analyze
  python process_ratcliff_lab_data.py --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze"
  
  # Process images_for_cellpose_model
  python process_ratcliff_lab_data.py --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_for_cellpose_model"
  
  # Process both and specify output directory
  python process_ratcliff_lab_data.py --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze" --output-dir ../datasets/RatcliffLab-analyze
  python process_ratcliff_lab_data.py --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_for_cellpose_model" --output-dir ../datasets/RatcliffLab-model
  
  # With weight maps
  python process_ratcliff_lab_data.py --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze" --create-weights
  
  # Different image size
  python process_ratcliff_lab_data.py --input-folder "../../images/ratcliff lab images/mitotic_spindle_segmentation_example/images_to_analyze" --target-size 1024
        """
    )
    
    parser.add_argument(
        '--input-folder', '-i',
        type=str,
        required=True,
        help='Input folder containing exported_images_ch1/ and exported_masks_raw/'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for dataset files (default: same as input folder name in ../datasets/)'
    )
    
    parser.add_argument(
        '--target-size', '-s',
        type=int,
        default=512,
        help='Target image size (default: 512)'
    )
    
    parser.add_argument(
        '--create-weights',
        action='store_true',
        help='Create weight maps for weighted loss'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize a sample after processing'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of samples to visualize (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        input_folder = Path(args.input_folder)
        output_dir = Path('../datasets') / input_folder.name
    
    print("="*70)
    print("SAMCell Dataset Processor - Ratcliff Lab Data")
    print("="*70)
    print(f"Input folder: {args.input_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Create weight maps: {args.create_weights}")
    
    # Process dataset
    imgs, dist_maps = process_dataset(
        args.input_folder,
        output_dir,
        target_size=args.target_size,
        create_weights=args.create_weights
    )
    
    # Visualize samples
    if args.visualize:
        import matplotlib.pyplot as plt
        
        print(f"\nVisualizing {args.num_samples} random samples...")
        n_samples = min(args.num_samples, len(imgs))
        indices = np.random.choice(len(imgs), n_samples, replace=False)
        
        for idx in indices:
            fig = visualize_sample(imgs, dist_maps, idx)
            plt.savefig(output_dir / f'sample_{idx}.png', dpi=150, bbox_inches='tight')
            print(f"  Saved visualization: {output_dir / f'sample_{idx}.png'}")
            plt.close()
    
    print("\n" + "="*70)
    print("✓ Processing complete!")
    print("="*70)
    print(f"\nYou can now train SAMCell with:")
    print(f"  python train.py --datasets {output_dir} --batch-size 4 --num-epochs 40")


if __name__ == '__main__':
    main()

