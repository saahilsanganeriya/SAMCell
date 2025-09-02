"""
Command-line interface for SAMCell.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import cv2
import numpy as np

from . import FinetunedSAM, SAMCellPipeline

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Load as grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    return image

def save_results(labels: np.ndarray, output_dir: str, base_name: str):
    """Save segmentation results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save labels as PNG
    labels_path = output_path / f"{base_name}_labels.png"
    # Normalize labels for visualization
    if np.max(labels) > 0:
        labels_vis = (labels / np.max(labels) * 255).astype(np.uint8)
    else:
        labels_vis = labels.astype(np.uint8)
    cv2.imwrite(str(labels_path), labels_vis)
    
    print(f"Segmentation saved to: {labels_path}")
    return labels_path

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SAMCell: Cell segmentation using Segment Anything",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic segmentation
  samcell segment image.png --model weights.pt --output results/
  
  # With metrics export
  samcell segment image.png --model weights.pt --output results/ --export-metrics
  
  # Custom thresholds
  samcell segment image.png --model weights.pt --peak-threshold 0.5 --fill-threshold 0.1
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment cells in an image')
    segment_parser.add_argument('image', help='Path to input image')
    segment_parser.add_argument('--model', '-m', required=True, 
                               help='Path to SAMCell model weights (.pt, .bin, or .safetensors)')
    segment_parser.add_argument('--output', '-o', default='./samcell_output',
                               help='Output directory (default: ./samcell_output)')
    segment_parser.add_argument('--peak-threshold', type=float, default=0.47,
                               help='Cell peak threshold (default: 0.47)')
    segment_parser.add_argument('--fill-threshold', type=float, default=0.09,
                               help='Cell fill threshold (default: 0.09)')
    segment_parser.add_argument('--crop-size', type=int, default=256,
                               help='Crop size for sliding window (default: 256)')
    segment_parser.add_argument('--export-metrics', action='store_true',
                               help='Export comprehensive metrics to CSV')
    segment_parser.add_argument('--include-texture', action='store_true',
                               help='Include texture metrics (slower)')
    segment_parser.add_argument('--device', default='auto',
                               help='Device to use (cuda/cpu/auto, default: auto)')
    segment_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Verbose output')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'version':
        from . import __version__
        print(f"SAMCell version {__version__}")
        return 0
    
    if args.command == 'segment':
        setup_logging(args.verbose)
        
        try:
            # Determine device
            if args.device == 'auto':
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = args.device
            
            print(f"Using device: {device}")
            
            # Load image
            print(f"Loading image: {args.image}")
            image = load_image(args.image)
            print(f"Image shape: {image.shape}")
            
            # Load model
            print(f"Loading model: {args.model}")
            model = FinetunedSAM('facebook/sam-vit-base')
            model.load_weights(args.model, map_location=device)
            
            # Create pipeline
            pipeline = SAMCellPipeline(model, device, crop_size=args.crop_size)
            pipeline.cells_max = args.peak_threshold
            pipeline.cell_fill = args.fill_threshold
            
            # Run segmentation
            print("Running segmentation...")
            labels = pipeline.run(image)
            
            # Count cells
            num_cells = len(np.unique(labels)) - 1  # -1 for background
            print(f"Found {num_cells} cells")
            
            # Save results
            base_name = Path(args.image).stem
            labels_path = save_results(labels, args.output, base_name)
            
            # Export metrics if requested
            if args.export_metrics:
                print("Calculating and exporting metrics...")
                metrics_path = Path(args.output) / f"{base_name}_metrics.csv"
                success = pipeline.export_metrics(
                    labels, str(metrics_path), image, args.include_texture
                )
                if success:
                    print(f"Metrics exported to: {metrics_path}")
                else:
                    print("Failed to export metrics")
            
            print("Segmentation complete!")
            return 0
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
