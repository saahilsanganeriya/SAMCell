"""
Dataset Processing Utilities for SAMCell
=========================================

Shared utilities for converting various dataset formats to SAMCell training format.

Output format:
- imgs.npy: Images (N, H, W) float32 array, normalized to [0, 1]
- dist_maps.npy: Distance maps (N, H, W) float32 array, normalized to [0, 1]
- wms.npy: Weight maps (N, H, W) float32 array (optional)
"""

import numpy as np
import cv2
from pathlib import Path

# Try to import scipy for better distance transforms
try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using OpenCV for distance transform")

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def preprocess_image(img):
    """
    Preprocess image with CLAHE normalization.

    Converts to grayscale, applies CLAHE for contrast enhancement,
    and converts to BGR for SAM compatibility.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W) or (H, W, C)

    Returns
    -------
    numpy.ndarray
        Preprocessed image (H, W, 3) uint8 in BGR format
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    # Normalize to 0-255 range
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_norm)

    # Convert to BGR for SAM processor
    img_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)

    return img_bgr


def create_distance_map(mask):
    """
    Create distance map from segmentation mask.

    Distance map shows the normalized Euclidean distance from each pixel
    to the nearest cell boundary. Interior of cells has positive distance.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask where each cell has a unique label (H, W)

    Returns
    -------
    numpy.ndarray
        Normalized distance map (H, W) float32 in [0, 1] range
    """
    # Create binary mask (cells vs background)
    binary_mask = (mask > 0).astype(np.uint8)

    # Compute distance transform
    if SCIPY_AVAILABLE:
        dist_map = distance_transform_edt(binary_mask)
    else:
        dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5).astype(np.float32)

    # Normalize to 0-1 range
    if dist_map.max() > 0:
        dist_map = dist_map / dist_map.max()

    return dist_map.astype(np.float32)


def create_weight_map(mask, w0=10, sigma=5):
    """
    Create weight map for weighted loss (emphasizes boundaries).

    Similar to U-Net paper, increases weight at cell boundaries
    to help separate touching cells.

    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask where each cell has a unique label
    w0 : float
        Weight of boundary regions (default: 10)
    sigma : float
        Standard deviation for Gaussian smoothing (default: 5)

    Returns
    -------
    numpy.ndarray
        Weight map (H, W) float32
    """
    # Find boundaries using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1)
    boundaries = (dilated - eroded) > 0

    # Create base weight map (1 everywhere)
    weight_map = np.ones_like(mask, dtype=np.float32)

    # Increase weight at boundaries
    weight_map[boundaries] = w0

    # Apply Gaussian smoothing
    weight_map = cv2.GaussianBlur(weight_map, (0, 0), sigma)

    return weight_map


def resize_with_padding(img, target_size=512, is_mask=False):
    """
    Resize image to target size while maintaining aspect ratio.
    Pads with zeros to make it square.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W) or (H, W, C)
    target_size : int
        Target size for both dimensions (default: 512)
    is_mask : bool
        Whether this is a mask (uses INTER_NEAREST) or image (uses INTER_LINEAR)

    Returns
    -------
    numpy.ndarray
        Resized and padded image (target_size, target_size)
    """
    h, w = img.shape[:2]

    # Calculate scaling factor to fit longest side to target_size
    if h > w:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)
    else:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)

    # Resize
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    # Pad to square
    pad_h = target_size - new_h
    pad_w = target_size - new_w

    img_padded = cv2.copyMakeBorder(
        img_resized, 0, pad_h, 0, pad_w,
        cv2.BORDER_CONSTANT, value=0
    )

    return img_padded


def normalize_image(img):
    """
    Normalize image to [0, 1] float32 range.

    Parameters
    ----------
    img : numpy.ndarray
        Input image

    Returns
    -------
    numpy.ndarray
        Normalized image (float32)
    """
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def save_dataset(imgs, dist_maps, output_dir, weight_maps=None):
    """
    Save processed dataset to .npy files.

    Parameters
    ----------
    imgs : numpy.ndarray
        Images array (N, H, W) or (N, H, W, 3)
    dist_maps : numpy.ndarray
        Distance maps array (N, H, W)
    output_dir : str or Path
        Output directory
    weight_maps : numpy.ndarray, optional
        Weight maps array (N, H, W)

    Returns
    -------
    dict
        Paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to proper format
    imgs = np.array(imgs, dtype=np.float32)
    dist_maps = np.array(dist_maps, dtype=np.float32)

    # Save files
    imgs_path = output_dir / 'imgs.npy'
    dist_maps_path = output_dir / 'dist_maps.npy'

    np.save(imgs_path, imgs)
    np.save(dist_maps_path, dist_maps)

    paths = {
        'imgs': str(imgs_path),
        'dist_maps': str(dist_maps_path)
    }

    if weight_maps is not None:
        weight_maps = np.array(weight_maps, dtype=np.float32)
        wms_path = output_dir / 'wms.npy'
        np.save(wms_path, weight_maps)
        paths['wms'] = str(wms_path)

    return paths


def print_dataset_stats(imgs, dist_maps, weight_maps=None):
    """
    Print statistics about the processed dataset.

    Parameters
    ----------
    imgs : numpy.ndarray
        Images array
    dist_maps : numpy.ndarray
        Distance maps array
    weight_maps : numpy.ndarray, optional
        Weight maps array
    """
    print("\nDataset Statistics:")
    print("=" * 60)
    print(f"Number of samples: {len(imgs)}")
    print(f"Image shape: {imgs.shape}")
    print(f"Distance map shape: {dist_maps.shape}")
    print(f"Image range: [{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"Distance map range: [{dist_maps.min():.3f}, {dist_maps.max():.3f}]")

    if weight_maps is not None:
        print(f"Weight map shape: {weight_maps.shape}")
        print(f"Weight map range: [{weight_maps.min():.3f}, {weight_maps.max():.3f}]")

    print("=" * 60)


def visualize_samples(imgs, dist_maps, output_dir, num_samples=3):
    """
    Visualize random samples from the dataset.

    Parameters
    ----------
    imgs : numpy.ndarray
        Images array
    dist_maps : numpy.ndarray
        Distance maps array
    output_dir : str or Path
        Output directory for saving visualizations
    num_samples : int
        Number of random samples to visualize (default: 3)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualization")
        return

    output_dir = Path(output_dir)
    n_samples = min(num_samples, len(imgs))
    indices = np.random.choice(len(imgs), n_samples, replace=False)

    for idx in indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        if len(imgs[idx].shape) == 3:
            axes[0].imshow(imgs[idx])
        else:
            axes[0].imshow(imgs[idx], cmap='gray')
        axes[0].set_title('Preprocessed Image')
        axes[0].axis('off')

        # Distance map
        axes[1].imshow(dist_maps[idx], cmap='jet')
        axes[1].set_title('Distance Map')
        axes[1].axis('off')

        # Overlay
        if len(imgs[idx].shape) == 3:
            axes[2].imshow(imgs[idx])
        else:
            axes[2].imshow(imgs[idx], cmap='gray')
        axes[2].imshow(dist_maps[idx], cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = output_dir / f'sample_{idx}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved visualization: {save_path}")
