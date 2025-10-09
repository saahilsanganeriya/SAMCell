#!/usr/bin/env python3
"""
SAMCell Dataset Processor
==========================

Unified script to convert various dataset formats to SAMCell training format.

Supports:
- LIVECell (COCO format with phase-contrast images)
- Cellpose (numbered image and mask pairs)
- Custom datasets (separate image and mask folders)

Output Format:
- imgs.npy: Preprocessed images (N, H, W, 3) float32, BGR format
- dist_maps.npy: Distance maps (N, H, W) float32, normalized [0, 1]
- wms.npy: Weight maps (N, H, W) float32 (optional)

Usage Examples:

    # LIVECell dataset (full training set)
    python process_dataset.py livecell \\
        --input /path/to/LIVECell_dataset_2021 \\
        --output ../datasets/LIVECell-train \\
        --split train

    # LIVECell 50% subset
    python process_dataset.py livecell \\
        --input /path/to/LIVECell_dataset_2021 \\
        --output ../datasets/LIVECell-train50 \\
        --split train_50pct

    # Cellpose dataset
    python process_dataset.py cellpose \\
        --input /path/to/CellPose/train_raw \\
        --output ../datasets/Cellpose-train \\
        --target-size 512

    # Custom mask dataset
    python process_dataset.py custom \\
        --images /path/to/images \\
        --masks /path/to/masks \\
        --output ../datasets/MyDataset \\
        --target-size 512
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

# Import utilities
from dataset_utils import (
    preprocess_image, create_distance_map, create_weight_map,
    resize_with_padding, normalize_image, save_dataset,
    print_dataset_stats, visualize_samples, tqdm
)


def process_livecell(base_folder, split, output_dir, create_weights=False):
    """
    Process LIVECell dataset in COCO format.

    Parameters
    ----------
    base_folder : str or Path
        Path to LIVECell_dataset_2021 folder
    split : str
        Dataset split: 'train', 'val', 'test', 'train_50pct', 'train_25pct', etc.
    output_dir : str or Path
        Output directory for processed dataset
    create_weights : bool
        Whether to create weight maps

    Returns
    -------
    tuple
        (imgs, dist_maps, weight_maps) numpy arrays
    """
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError("pycocotools required for LIVECell. Install with: pip install pycocotools")

    base_folder = Path(base_folder)
    output_dir = Path(output_dir)

    # Map split names to annotation files
    split_map = {
        'train': 'annotations/LIVECell/livecell_coco_train.json',
        'val': 'annotations/LIVECell/livecell_coco_val.json',
        'test': 'annotations/LIVECell/livecell_coco_test.json',
        'train_50pct': 'annotations/LIVECell_dataset_size_split/4_train50percent.json',
        'train_25pct': 'annotations/LIVECell_dataset_size_split/3_train25percent.json',
        'train_5pct': 'annotations/LIVECell_dataset_size_split/2_train5percent.json',
        'train_4pct': 'annotations/LIVECell_dataset_size_split/1_train4percent.json',
        'train_2pct': 'annotations/LIVECell_dataset_size_split/0_train2percent.json',
    }

    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Options: {list(split_map.keys())}")

    ann_file = base_folder / split_map[split]
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")

    # Determine image folder
    if split == 'test':
        img_folder = base_folder / 'images' / 'livecell_test_images'
    else:
        img_folder = base_folder / 'images' / 'livecell_train_val_images'

    print(f"\nProcessing LIVECell dataset ({split} split)...")
    print(f"Annotation file: {ann_file}")
    print(f"Image folder: {img_folder}")

    # Load COCO dataset
    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()
    imgs_meta = coco.loadImgs(img_ids)

    print(f"Found {len(imgs_meta)} images")

    imgs = []
    dist_maps = []
    weight_maps = [] if create_weights else None

    for img_meta in tqdm(imgs_meta, desc="Processing images"):
        # Load image
        img_path = img_folder / img_meta['file_name']
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"Warning: Could not load {img_path}, skipping")
            continue

        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_meta['id'])
        anns = coco.loadAnns(ann_ids)

        # Create segmentation mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.int16)

        cell_id = 1
        for ann in anns:
            if ann['iscrowd'] != 0:
                # Skip crowd annotations
                continue

            # Convert annotation to mask
            cell_mask = coco.annToMask(ann)
            mask[cell_mask == 1] = cell_id
            cell_id += 1

        # Preprocess image
        img_processed = preprocess_image(img)

        # Create distance map
        dist_map = create_distance_map(mask)

        # Normalize image
        img_normalized = normalize_image(img_processed)

        imgs.append(img_normalized)
        dist_maps.append(dist_map)

        if create_weights:
            weight_map = create_weight_map(mask)
            weight_maps.append(weight_map)

    # Convert to arrays
    imgs = np.array(imgs, dtype=np.float32)
    dist_maps = np.array(dist_maps, dtype=np.float32)
    if create_weights:
        weight_maps = np.array(weight_maps, dtype=np.float32)

    # Save dataset
    paths = save_dataset(imgs, dist_maps, output_dir, weight_maps)
    print(f"\n✓ Saved to {output_dir}:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    print_dataset_stats(imgs, dist_maps, weight_maps)

    return imgs, dist_maps, weight_maps


def process_cellpose(input_folder, output_dir, img_pattern='{:03d}_img.png',
                     mask_pattern='{:03d}_masks.png', target_size=512,
                     create_weights=False):
    """
    Process Cellpose dataset with numbered image/mask pairs.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing numbered images and masks
    output_dir : str or Path
        Output directory
    img_pattern : str
        Filename pattern for images (default: '{:03d}_img.png')
    mask_pattern : str
        Filename pattern for masks (default: '{:03d}_masks.png')
    target_size : int
        Target image size (default: 512)
    create_weights : bool
        Whether to create weight maps

    Returns
    -------
    tuple
        (imgs, dist_maps, weight_maps) numpy arrays
    """
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)

    print(f"\nProcessing Cellpose dataset...")
    print(f"Input folder: {input_folder}")
    print(f"Target size: {target_size}x{target_size}")

    imgs = []
    dist_maps = []
    weight_maps = [] if create_weights else None

    # Find all image-mask pairs
    i = 0
    while True:
        img_file = input_folder / img_pattern.format(i)
        mask_file = input_folder / mask_pattern.format(i)

        if not img_file.exists() or not mask_file.exists():
            break

        # Load image and mask
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

        if img is None or mask is None:
            print(f"Warning: Could not load pair {i}, skipping")
            i += 1
            continue

        # Resize to target size (maintaining aspect ratio + padding)
        img_resized = resize_with_padding(img, target_size, is_mask=False)
        mask_resized = resize_with_padding(mask, target_size, is_mask=True)

        # Preprocess image
        img_processed = preprocess_image(img_resized)

        # Create distance map
        dist_map = create_distance_map(mask_resized)

        # Normalize image
        img_normalized = normalize_image(img_processed)

        imgs.append(img_normalized)
        dist_maps.append(dist_map)

        if create_weights:
            weight_map = create_weight_map(mask_resized)
            weight_maps.append(weight_map)

        i += 1

    print(f"Found {i} image-mask pairs")

    if i == 0:
        raise ValueError(f"No image-mask pairs found in {input_folder}")

    # Convert to arrays
    imgs = np.array(imgs, dtype=np.float32)
    dist_maps = np.array(dist_maps, dtype=np.float32)
    if create_weights:
        weight_maps = np.array(weight_maps, dtype=np.float32)

    # Save dataset
    paths = save_dataset(imgs, dist_maps, output_dir, weight_maps)
    print(f"\n✓ Saved to {output_dir}:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    print_dataset_stats(imgs, dist_maps, weight_maps)

    return imgs, dist_maps, weight_maps


def process_custom(image_folder, mask_folder, output_dir, image_ext='.png',
                   mask_suffix='_mask', target_size=512, create_weights=False):
    """
    Process custom dataset with separate image and mask folders.

    Parameters
    ----------
    image_folder : str or Path
        Folder containing images
    mask_folder : str or Path
        Folder containing masks
    output_dir : str or Path
        Output directory
    image_ext : str
        Image file extension (default: '.png')
    mask_suffix : str
        Suffix added to image name for mask filename (default: '_mask')
    target_size : int
        Target image size (default: 512)
    create_weights : bool
        Whether to create weight maps

    Returns
    -------
    tuple
        (imgs, dist_maps, weight_maps) numpy arrays
    """
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    output_dir = Path(output_dir)

    print(f"\nProcessing custom dataset...")
    print(f"Image folder: {image_folder}")
    print(f"Mask folder: {mask_folder}")
    print(f"Target size: {target_size}x{target_size}")

    # Find all images
    img_files = sorted(list(image_folder.glob(f'*{image_ext}')))

    if len(img_files) == 0:
        raise ValueError(f"No images found in {image_folder} with extension {image_ext}")

    print(f"Found {len(img_files)} images")

    imgs = []
    dist_maps = []
    weight_maps = [] if create_weights else None

    for img_file in tqdm(img_files, desc="Processing images"):
        # Find corresponding mask
        img_stem = img_file.stem
        mask_file = mask_folder / f"{img_stem}{mask_suffix}{image_ext}"

        if not mask_file.exists():
            print(f"Warning: No mask found for {img_file.name}, skipping")
            continue

        # Load image and mask
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)

        if img is None or mask is None:
            print(f"Warning: Could not load {img_file.name}, skipping")
            continue

        # Verify dimensions match
        if img.shape[:2] != mask.shape[:2]:
            print(f"Warning: Dimension mismatch for {img_file.name}, skipping")
            continue

        # Resize
        img_resized = resize_with_padding(img, target_size, is_mask=False)
        mask_resized = resize_with_padding(mask, target_size, is_mask=True)

        # Preprocess
        img_processed = preprocess_image(img_resized)

        # Create distance map
        dist_map = create_distance_map(mask_resized)

        # Normalize
        img_normalized = normalize_image(img_processed)

        imgs.append(img_normalized)
        dist_maps.append(dist_map)

        if create_weights:
            weight_map = create_weight_map(mask_resized)
            weight_maps.append(weight_map)

    if len(imgs) == 0:
        raise ValueError("No valid image-mask pairs found")

    # Convert to arrays
    imgs = np.array(imgs, dtype=np.float32)
    dist_maps = np.array(dist_maps, dtype=np.float32)
    if create_weights:
        weight_maps = np.array(weight_maps, dtype=np.float32)

    # Save dataset
    paths = save_dataset(imgs, dist_maps, output_dir, weight_maps)
    print(f"\n✓ Saved to {output_dir}:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    print_dataset_stats(imgs, dist_maps, weight_maps)

    return imgs, dist_maps, weight_maps


def main():
    parser = argparse.ArgumentParser(
        description='Convert datasets to SAMCell training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='dataset_type', help='Dataset type')
    subparsers.required = True

    # ==== LIVECell dataset ====
    livecell_parser = subparsers.add_parser(
        'livecell',
        help='Process LIVECell dataset (COCO format)'
    )
    livecell_parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Path to LIVECell_dataset_2021 folder'
    )
    livecell_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output directory for processed dataset'
    )
    livecell_parser.add_argument(
        '--split', type=str,
        choices=['train', 'val', 'test', 'train_50pct', 'train_25pct',
                 'train_5pct', 'train_4pct', 'train_2pct'],
        default='train_50pct',
        help='Dataset split to process (default: train_50pct)'
    )
    livecell_parser.add_argument(
        '--create-weights', action='store_true',
        help='Create weight maps for boundary emphasis'
    )
    livecell_parser.add_argument(
        '--visualize', action='store_true',
        help='Generate sample visualizations'
    )

    # ==== Cellpose dataset ====
    cellpose_parser = subparsers.add_parser(
        'cellpose',
        help='Process Cellpose dataset (numbered image and mask pairs)'
    )
    cellpose_parser.add_argument(
        '--input', '-i', type=str, required=True,
        help='Path to folder containing numbered images and masks'
    )
    cellpose_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output directory for processed dataset'
    )
    cellpose_parser.add_argument(
        '--img-pattern', type=str, default='{:03d}_img.png',
        help='Image filename pattern (default: {:03d}_img.png)'
    )
    cellpose_parser.add_argument(
        '--mask-pattern', type=str, default='{:03d}_masks.png',
        help='Mask filename pattern (default: {:03d}_masks.png)'
    )
    cellpose_parser.add_argument(
        '--target-size', type=int, default=512,
        help='Resize images to this size (default: 512)'
    )
    cellpose_parser.add_argument(
        '--create-weights', action='store_true',
        help='Create weight maps for boundary emphasis'
    )
    cellpose_parser.add_argument(
        '--visualize', action='store_true',
        help='Generate sample visualizations'
    )

    # ==== Custom mask dataset ====
    custom_parser = subparsers.add_parser(
        'custom',
        help='Process custom dataset with separate image and mask folders'
    )
    custom_parser.add_argument(
        '--images', type=str, required=True,
        help='Path to folder containing images'
    )
    custom_parser.add_argument(
        '--masks', type=str, required=True,
        help='Path to folder containing masks'
    )
    custom_parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output directory for processed dataset'
    )
    custom_parser.add_argument(
        '--image-ext', type=str, default='.png',
        help='Image file extension (default: .png)'
    )
    custom_parser.add_argument(
        '--mask-suffix', type=str, default='_mask',
        help='Mask filename suffix (default: _mask)'
    )
    custom_parser.add_argument(
        '--target-size', type=int, default=512,
        help='Resize images to this size (default: 512)'
    )
    custom_parser.add_argument(
        '--create-weights', action='store_true',
        help='Create weight maps for boundary emphasis'
    )
    custom_parser.add_argument(
        '--visualize', action='store_true',
        help='Generate sample visualizations'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SAMCell Dataset Processor")
    print("=" * 70)
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output directory: {args.output}")

    # Process based on dataset type
    if args.dataset_type == 'livecell':
        imgs, dist_maps, weight_maps = process_livecell(
            args.input, args.split, args.output, args.create_weights
        )

    elif args.dataset_type == 'cellpose':
        imgs, dist_maps, weight_maps = process_cellpose(
            args.input, args.output, args.img_pattern, args.mask_pattern,
            args.target_size, args.create_weights
        )

    elif args.dataset_type == 'custom':
        imgs, dist_maps, weight_maps = process_custom(
            args.images, args.masks, args.output,
            args.image_ext, args.mask_suffix, args.target_size, args.create_weights
        )

    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_samples(imgs, dist_maps, args.output, num_samples=3)

    print("\n" + "=" * 70)
    print("✓ Processing complete!")
    print("=" * 70)
    print(f"\nDataset ready for training with {len(imgs)} samples")
    print(f"\nTo train SAMCell:")
    print(f"  cd ../training")
    print(f"  python train.py --datasets {args.output} --batch-size 4 --num-epochs 40")


if __name__ == '__main__':
    main()
