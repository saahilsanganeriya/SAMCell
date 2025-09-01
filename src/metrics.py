"""
Comprehensive cell shape and morphology metrics for SAMCell segmentation results.

This module provides functions to calculate various morphological and intensity-based
metrics from segmented cell images, based on the SAMCell paper and common cell analysis workflows.

Author: SAMCell Team
"""

import numpy as np
import cv2
import pandas as pd
from skimage import measure, morphology
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage
from scipy.spatial.distance import cdist
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings

logger = logging.getLogger(__name__)

def calculate_basic_metrics(labels: np.ndarray, original_image: Optional[np.ndarray] = None) -> Dict[str, List[float]]:
    """
    Calculate basic morphological metrics for each segmented cell.
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask where each unique non-zero value represents a cell
    original_image : np.ndarray, optional
        Original grayscale image for intensity-based metrics
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary containing lists of metric values for each cell
    """
    try:
        # Get unique labels (excluding background)
        unique_labels = np.unique(labels[labels > 0])
        
        if len(unique_labels) == 0:
            logger.warning("No cells found in segmentation")
            return {}
            
        # Initialize metric containers
        metrics = {
            'cell_id': [],
            'area': [],
            'perimeter': [],
            'convex_area': [],
            'convex_perimeter': [],
            'centroid_x': [],
            'centroid_y': [],
            'bbox_min_row': [],
            'bbox_min_col': [],
            'bbox_max_row': [],
            'bbox_max_col': [],
            'equivalent_diameter': [],
            'major_axis_length': [],
            'minor_axis_length': [],
            'eccentricity': [],
            'orientation': [],
            'solidity': [],
            'extent': [],
            'compactness': [],
            'circularity': [],
            'aspect_ratio': [],
            'roundness': [],
        }
        
        # Add intensity metrics if original image is provided
        if original_image is not None:
            metrics.update({
                'mean_intensity': [],
                'std_intensity': [],
                'min_intensity': [],
                'max_intensity': [],
                'intensity_range': [],
            })
        
        # Calculate metrics for each cell
        for label_val in unique_labels:
            try:
                # Create binary mask for this cell
                cell_mask = (labels == label_val).astype(np.uint8)
                
                # Get region properties
                props = measure.regionprops(cell_mask, intensity_image=original_image)[0]
                
                # Basic geometric properties
                metrics['cell_id'].append(int(label_val))
                metrics['area'].append(float(props.area))
                metrics['perimeter'].append(float(props.perimeter))
                metrics['convex_area'].append(float(props.convex_area))
                
                # Convex perimeter
                convex_perimeter = measure.perimeter(props.convex_image)
                metrics['convex_perimeter'].append(float(convex_perimeter))
                
                # Centroid
                centroid = props.centroid
                metrics['centroid_x'].append(float(centroid[1]))  # Note: switched for x,y convention
                metrics['centroid_y'].append(float(centroid[0]))
                
                # Bounding box
                bbox = props.bbox
                metrics['bbox_min_row'].append(float(bbox[0]))
                metrics['bbox_min_col'].append(float(bbox[1]))
                metrics['bbox_max_row'].append(float(bbox[2]))
                metrics['bbox_max_col'].append(float(bbox[3]))
                
                # Shape descriptors
                metrics['equivalent_diameter'].append(float(props.equivalent_diameter))
                metrics['major_axis_length'].append(float(props.major_axis_length))
                metrics['minor_axis_length'].append(float(props.minor_axis_length))
                metrics['eccentricity'].append(float(props.eccentricity))
                metrics['orientation'].append(float(props.orientation))
                metrics['solidity'].append(float(props.solidity))
                metrics['extent'].append(float(props.extent))
                
                # Derived shape metrics
                area = props.area
                perimeter = props.perimeter
                
                # Compactness (also called form factor)
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                metrics['compactness'].append(float(compactness))
                
                # Circularity (another definition)
                circularity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
                metrics['circularity'].append(float(circularity))
                
                # Aspect ratio
                aspect_ratio = props.major_axis_length / props.minor_axis_length if props.minor_axis_length > 0 else 0
                metrics['aspect_ratio'].append(float(aspect_ratio))
                
                # Roundness
                roundness = (4 * area) / (np.pi * props.major_axis_length ** 2) if props.major_axis_length > 0 else 0
                metrics['roundness'].append(float(roundness))
                
                # Intensity metrics
                if original_image is not None:
                    metrics['mean_intensity'].append(float(props.mean_intensity))
                    metrics['std_intensity'].append(float(np.std(original_image[cell_mask > 0])))
                    metrics['min_intensity'].append(float(props.min_intensity))
                    metrics['max_intensity'].append(float(props.max_intensity))
                    metrics['intensity_range'].append(float(props.max_intensity - props.min_intensity))
                    
            except Exception as e:
                logger.error(f"Error calculating metrics for cell {label_val}: {str(e)}")
                continue
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_basic_metrics: {str(e)}")
        return {}

def calculate_neighbor_metrics(labels: np.ndarray, neighbor_distance: int = 10) -> Dict[str, List[float]]:
    """
    Calculate neighbor-based metrics for each cell.
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask
    neighbor_distance : int
        Distance in pixels to consider for neighbors
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary containing neighbor metrics for each cell
    """
    try:
        unique_labels = np.unique(labels[labels > 0])
        
        if len(unique_labels) == 0:
            return {}
            
        metrics = {
            'cell_id': [],
            'num_neighbors': [],
            'avg_neighbor_distance': [],
            'min_neighbor_distance': [],
            'nearest_neighbor_distance': [],
        }
        
        # Get centroids for all cells
        props_list = measure.regionprops(labels)
        centroids = {prop.label: prop.centroid for prop in props_list}
        
        for label_val in unique_labels:
            try:
                metrics['cell_id'].append(int(label_val))
                
                if label_val not in centroids:
                    # Fill with default values if cell not found
                    metrics['num_neighbors'].append(0)
                    metrics['avg_neighbor_distance'].append(0.0)
                    metrics['min_neighbor_distance'].append(0.0)
                    metrics['nearest_neighbor_distance'].append(0.0)
                    continue
                
                current_centroid = centroids[label_val]
                
                # Calculate distances to all other cells
                distances = []
                neighbors_within_distance = 0
                
                for other_label, other_centroid in centroids.items():
                    if other_label != label_val:
                        dist = np.sqrt((current_centroid[0] - other_centroid[0])**2 + 
                                     (current_centroid[1] - other_centroid[1])**2)
                        distances.append(dist)
                        if dist <= neighbor_distance:
                            neighbors_within_distance += 1
                
                metrics['num_neighbors'].append(neighbors_within_distance)
                
                if distances:
                    metrics['avg_neighbor_distance'].append(float(np.mean(distances)))
                    metrics['min_neighbor_distance'].append(float(np.min(distances)))
                    metrics['nearest_neighbor_distance'].append(float(np.min(distances)))
                else:
                    metrics['avg_neighbor_distance'].append(0.0)
                    metrics['min_neighbor_distance'].append(0.0)
                    metrics['nearest_neighbor_distance'].append(0.0)
                    
            except Exception as e:
                logger.error(f"Error calculating neighbor metrics for cell {label_val}: {str(e)}")
                continue
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_neighbor_metrics: {str(e)}")
        return {}

def calculate_texture_metrics(labels: np.ndarray, original_image: np.ndarray, 
                            distances: List[int] = [1, 2], angles: List[float] = [0, 45, 90, 135]) -> Dict[str, List[float]]:
    """
    Calculate texture metrics using Gray Level Co-occurrence Matrix (GLCM).
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask
    original_image : np.ndarray
        Original grayscale image
    distances : List[int]
        Distances for GLCM calculation
    angles : List[float]
        Angles for GLCM calculation (in degrees)
        
    Returns
    -------
    Dict[str, List[float]]
        Dictionary containing texture metrics for each cell
    """
    try:
        unique_labels = np.unique(labels[labels > 0])
        
        if len(unique_labels) == 0:
            return {}
            
        # Convert angles to radians
        angles_rad = [np.deg2rad(angle) for angle in angles]
        
        metrics = {
            'cell_id': [],
            'contrast_mean': [],
            'dissimilarity_mean': [],
            'homogeneity_mean': [],
            'energy_mean': [],
            'correlation_mean': [],
            'asm_mean': [],  # Angular Second Moment
        }
        
        for label_val in unique_labels:
            try:
                metrics['cell_id'].append(int(label_val))
                
                # Create binary mask for this cell
                cell_mask = (labels == label_val)
                
                # Extract cell region from original image
                cell_region = original_image[cell_mask]
                
                if len(cell_region) < 4:  # Need minimum pixels for GLCM
                    # Fill with default values for very small cells
                    for key in ['contrast_mean', 'dissimilarity_mean', 'homogeneity_mean', 
                              'energy_mean', 'correlation_mean', 'asm_mean']:
                        metrics[key].append(0.0)
                    continue
                
                # Get bounding box to crop the region
                props = measure.regionprops(labels == label_val)[0]
                bbox = props.bbox
                cropped_image = original_image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                cropped_mask = (labels[bbox[0]:bbox[2], bbox[1]:bbox[3]] == label_val)
                
                # Mask the cropped image
                masked_image = cropped_image.copy()
                masked_image[~cropped_mask] = 0
                
                # Calculate GLCM properties
                contrast_values = []
                dissimilarity_values = []
                homogeneity_values = []
                energy_values = []
                correlation_values = []
                asm_values = []
                
                for distance in distances:
                    try:
                        # Calculate GLCM
                        glcm = graycomatrix(masked_image, distances=[distance], angles=angles_rad, 
                                         levels=256, symmetric=True, normed=True)
                        
                        # Calculate properties
                        contrast_values.extend(graycoprops(glcm, 'contrast').flatten())
                        dissimilarity_values.extend(graycoprops(glcm, 'dissimilarity').flatten())
                        homogeneity_values.extend(graycoprops(glcm, 'homogeneity').flatten())
                        energy_values.extend(graycoprops(glcm, 'energy').flatten())
                        correlation_values.extend(graycoprops(glcm, 'correlation').flatten())
                        asm_values.extend(graycoprops(glcm, 'ASM').flatten())
                        
                    except Exception as e:
                        logger.warning(f"Error calculating GLCM for cell {label_val}, distance {distance}: {str(e)}")
                        continue
                
                # Store mean values
                metrics['contrast_mean'].append(float(np.mean(contrast_values)) if contrast_values else 0.0)
                metrics['dissimilarity_mean'].append(float(np.mean(dissimilarity_values)) if dissimilarity_values else 0.0)
                metrics['homogeneity_mean'].append(float(np.mean(homogeneity_values)) if homogeneity_values else 0.0)
                metrics['energy_mean'].append(float(np.mean(energy_values)) if energy_values else 0.0)
                metrics['correlation_mean'].append(float(np.mean(correlation_values)) if correlation_values else 0.0)
                metrics['asm_mean'].append(float(np.mean(asm_values)) if asm_values else 0.0)
                
            except Exception as e:
                logger.error(f"Error calculating texture metrics for cell {label_val}: {str(e)}")
                continue
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_texture_metrics: {str(e)}")
        return {}

def calculate_summary_metrics(labels: np.ndarray, original_image: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate summary metrics for the entire image/field of view.
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask
    original_image : np.ndarray, optional
        Original grayscale image
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing summary metrics
    """
    try:
        unique_labels = np.unique(labels[labels > 0])
        
        metrics = {
            'total_cells': len(unique_labels),
            'total_cell_area': float(np.sum(labels > 0)),
            'image_area': float(labels.size),
            'confluency': 0.0,
            'avg_cell_area': 0.0,
            'std_cell_area': 0.0,
            'avg_cell_perimeter': 0.0,
            'cell_density': 0.0,  # cells per unit area
        }
        
        if len(unique_labels) == 0:
            return metrics
            
        # Calculate confluency (percentage of image covered by cells)
        metrics['confluency'] = float(metrics['total_cell_area'] / metrics['image_area'] * 100)
        
        # Calculate per-cell metrics
        cell_areas = []
        cell_perimeters = []
        
        for label_val in unique_labels:
            cell_mask = (labels == label_val)
            area = np.sum(cell_mask)
            cell_areas.append(area)
            
            # Calculate perimeter
            props = measure.regionprops(cell_mask.astype(int))[0]
            cell_perimeters.append(props.perimeter)
        
        if cell_areas:
            metrics['avg_cell_area'] = float(np.mean(cell_areas))
            metrics['std_cell_area'] = float(np.std(cell_areas))
            
        if cell_perimeters:
            metrics['avg_cell_perimeter'] = float(np.mean(cell_perimeters))
            
        # Cell density (cells per 1000 pixels)
        metrics['cell_density'] = float(len(unique_labels) / (metrics['image_area'] / 1000))
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in calculate_summary_metrics: {str(e)}")
        return {}

def calculate_all_metrics(labels: np.ndarray, original_image: Optional[np.ndarray] = None, 
                         include_texture: bool = False, neighbor_distance: int = 10) -> pd.DataFrame:
    """
    Calculate all available metrics and return as a pandas DataFrame.
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask where each unique non-zero value represents a cell
    original_image : np.ndarray, optional
        Original grayscale image for intensity-based metrics
    include_texture : bool
        Whether to include texture metrics (computationally expensive)
    neighbor_distance : int
        Distance in pixels to consider for neighbor analysis
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all metrics for each cell
    """
    try:
        logger.info("Calculating comprehensive cell metrics...")
        
        # Calculate basic metrics
        basic_metrics = calculate_basic_metrics(labels, original_image)
        if not basic_metrics:
            logger.warning("No basic metrics calculated")
            return pd.DataFrame()
            
        # Calculate neighbor metrics
        neighbor_metrics = calculate_neighbor_metrics(labels, neighbor_distance)
        
        # Combine metrics
        all_metrics = basic_metrics.copy()
        
        # Add neighbor metrics (matching by cell_id)
        if neighbor_metrics and 'cell_id' in neighbor_metrics:
            neighbor_df = pd.DataFrame(neighbor_metrics)
            basic_df = pd.DataFrame(basic_metrics)
            
            # Merge on cell_id
            combined_df = pd.merge(basic_df, neighbor_df, on='cell_id', how='left')
            all_metrics = combined_df.to_dict('list')
        
        # Calculate texture metrics if requested
        if include_texture and original_image is not None:
            try:
                texture_metrics = calculate_texture_metrics(labels, original_image)
                if texture_metrics and 'cell_id' in texture_metrics:
                    texture_df = pd.DataFrame(texture_metrics)
                    current_df = pd.DataFrame(all_metrics)
                    combined_df = pd.merge(current_df, texture_df, on='cell_id', how='left')
                    all_metrics = combined_df.to_dict('list')
            except Exception as e:
                logger.warning(f"Error calculating texture metrics, skipping: {str(e)}")
        
        # Create final DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Sort by cell_id
        if 'cell_id' in df.columns:
            df = df.sort_values('cell_id').reset_index(drop=True)
        
        logger.info(f"Calculated metrics for {len(df)} cells with {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"Error in calculate_all_metrics: {str(e)}")
        return pd.DataFrame()

def export_metrics_csv(labels: np.ndarray, output_path: str, original_image: Optional[np.ndarray] = None,
                      include_texture: bool = False, include_summary: bool = True) -> bool:
    """
    Calculate metrics and export to CSV file.
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask
    output_path : str
        Path to save the CSV file
    original_image : np.ndarray, optional
        Original grayscale image
    include_texture : bool
        Whether to include texture metrics
    include_summary : bool
        Whether to include summary metrics as additional rows
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Calculate all metrics
        df = calculate_all_metrics(labels, original_image, include_texture)
        
        if df.empty:
            logger.error("No metrics calculated, cannot export CSV")
            return False
        
        # Add summary metrics if requested
        if include_summary:
            summary = calculate_summary_metrics(labels, original_image)
            
            # Add summary as additional rows
            summary_rows = []
            for key, value in summary.items():
                summary_row = {'metric_type': 'summary', 'metric_name': key, 'value': value}
                # Fill other columns with NaN
                for col in df.columns:
                    if col not in summary_row:
                        summary_row[col] = np.nan
                summary_rows.append(summary_row)
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                # Add a separator row
                separator_row = pd.DataFrame([{col: '---' if col in ['metric_type', 'metric_name'] else np.nan 
                                             for col in summary_df.columns}])
                df = pd.concat([df, separator_row, summary_df], ignore_index=True)
        
        # Export to CSV
        df.to_csv(output_path, index=False)
        logger.info(f"Metrics exported to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting metrics to CSV: {str(e)}")
        return False

# Convenience function for the GUI
def compute_gui_metrics(labels: np.ndarray, original_image: Optional[np.ndarray] = None) -> Tuple[int, float, str, float]:
    """
    Compute the basic metrics shown in the original GUI.
    
    Parameters
    ----------
    labels : np.ndarray
        Labeled segmentation mask
    original_image : np.ndarray, optional
        Original grayscale image
        
    Returns
    -------
    Tuple[int, float, str, float]
        cell_count, avg_cell_area, confluency_str, avg_neighbors
    """
    try:
        # Cell count
        unique_labels = np.unique(labels[labels > 0])
        cell_count = len(unique_labels)
        
        if cell_count == 0:
            return 0, 0.0, "0%", 0.0
        
        # Cell area
        total_cell_area = np.sum(labels != 0)
        avg_cell_area = total_cell_area / cell_count
        
        # Confluency
        confluency = total_cell_area / (labels.shape[0] * labels.shape[1])
        confluency_str = f'{int(confluency * 100)}%'
        
        # Average neighbors (using the original algorithm from GUI)
        neighbors = []
        for label_val in unique_labels:
            cell_coords = labels == label_val
            # Add 5 pixel buffer around cell
            cell_coords = np.where(cell_coords)
            cell_coords = (np.clip(cell_coords[0] - 5, 0, labels.shape[0] - 1), 
                          np.clip(cell_coords[1] - 5, 0, labels.shape[1] - 1))
            
            # Get all cells within buffer
            neighbor_cells = np.unique(labels[cell_coords[0], cell_coords[1]])
            neighbors.append(len(neighbor_cells) - 1)  # -1 to exclude self
        
        avg_neighbors = np.mean(neighbors) if neighbors else 0.0
        
        return cell_count, round(avg_cell_area, 2), confluency_str, round(avg_neighbors, 2)
        
    except Exception as e:
        logger.error(f"Error in compute_gui_metrics: {str(e)}")
        return 0, 0.0, "0%", 0.0
