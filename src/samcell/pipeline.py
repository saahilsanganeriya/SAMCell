from typing import Any, Optional, Callable, Dict, Tuple, Union, List
from skimage.segmentation import watershed
import math
import torch
from torch import nn
import numpy as np
import cv2
from transformers import SamProcessor
from .slidingWindow import SlidingWindowHelper
from skimage import measure
from tqdm import tqdm
import logging
import traceback
from .metrics import calculate_all_metrics, export_metrics_csv, compute_gui_metrics, calculate_metric_statistics, export_metrics_excel

logger = logging.getLogger(__name__)


class SlidingWindowPipeline:
    def __init__(self, model, device, crop_size=256, cells_max=0.47, cell_fill=0.09):
        try:
            logger.info("Initializing SlidingWindowPipeline")
            self.model = model.get_model()
            
            self.device = device
            self.crop_size = crop_size
            self.sigmoid = nn.Sigmoid()
            self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
            self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)
            
            # Default thresholds - updated to match napari
            self.cells_max_threshold = cells_max
            self.cell_fill_threshold = cell_fill
            
            logger.info("SlidingWindowPipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _preprocess(self, img):
        try:
            # Ensure image is not empty
            if img is None or img.size == 0:
                raise ValueError("Input image is empty")
                
            # Make sure image has the right type
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            # Apply CLAHE for contrast enhancement
            img = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Convert to color if necessary
            if len(img.shape) != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # Default SAM preprocessing
            inputs = self.processor(img, return_tensors="pt")
            return inputs['pixel_values'].to(self.device)
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _preprocess_sam(self, img):
        inputs = self.processor(img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def get_model_prediction(self, image):
        try:
            image_orig = image.copy()
            image = self._preprocess(image_orig)
            self.model.eval().to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs_finetuned = self.model(pixel_values=image, multimask_output=True)

            prob_finetuned = outputs_finetuned['pred_masks'].squeeze(1)

            # Sigmoid
            dist_map = self.sigmoid(prob_finetuned)[0][0]

            return dist_map
        except Exception as e:
            logger.error(f"Error in get_model_prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def spilt_into_crops(self, image_orig):
        crops = []
        # split into 512x512 crops
        for i in range(0, math.ceil(image_orig.shape[0] / (self.crop_size)) + 1):
            for j in range(0, math.ceil(image_orig.shape[1] / self.crop_size) + 1):
                min_x = i * self.crop_size
                min_y = j * self.crop_size
                min_x = min(min_x, image_orig.shape[0] - self.crop_size)
                min_y = min(min_y, image_orig.shape[1] - self.crop_size)
                crops.append((image_orig[min_x:min_x + self.crop_size, min_y:min_y + self.crop_size], (min_x, min_y)))

        return crops

    def predict_on_full_img(self, image_orig):
        orig_shape = image_orig.shape
        # rescale to 1000 (biggest dimension)
        if image_orig.shape[0] > image_orig.shape[1]:
            new_size = (int(image_orig.shape[1] * (1000 / image_orig.shape[0])), 1000)
        else:
            new_size = (1000, int(image_orig.shape[0] * (1000 / image_orig.shape[1])))

        # image_orig = cv2.resize(image_orig, (new_size))

        # crops = self.spilt_into_crops(image_orig)
        crops, orig_regions, crop_unique_region = self.sliding_window_helper.seperate_into_crops(image_orig)

        # predict on crops
        dist_maps = []
        for crop in crops:
            dist_map = self.get_model_prediction(crop).cpu().numpy()

            # resize to crop size
            dist_map = cv2.resize(dist_map, (self.crop_size, self.crop_size))
            dist_maps.append(dist_map)

        # reconstruct image
        # cell_dist_map = np.zeros(image_orig.shape)
        # for crop in crops:
        #     min_x,min_y = crop[1]
        #     cell_dist_map[min_x:min_x+self.crop_size, min_y:min_y+self.crop_size] = dist_maps.pop(0)

        cell_dist_map = self.sliding_window_helper.combine_crops(orig_shape, crops, orig_regions, crop_unique_region,
                                                                 dist_maps)

        return cell_dist_map

    def cells_from_dist_map(self, dist_map, cells_max_threshold=None, cell_fill_threshold=None):
        try:
            # Use provided thresholds or defaults
            cells_max_threshold = cells_max_threshold if cells_max_threshold is not None else self.cells_max_threshold
            cell_fill_threshold = cell_fill_threshold if cell_fill_threshold is not None else self.cell_fill_threshold
            
            # Apply thresholds
            cells_max = dist_map > cells_max_threshold
            cell_fill = dist_map > cell_fill_threshold
            
            # Debug log
            logger.info(f"cells_max threshold: {cells_max_threshold}, sum: {np.sum(cells_max)}")
            logger.info(f"cell_fill threshold: {cell_fill_threshold}, sum: {np.sum(cell_fill)}")
            
            # Convert to binary masks
            cells_max = cells_max.astype(np.uint8)
            cell_fill = cell_fill.astype(np.uint8)
            
            # Find contours - handle different OpenCV versions
            try:
                contours, _ = cv2.findContours(cells_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours, _ = cv2.findContours(cells_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
            logger.info(f"Found {len(contours)} contours")
                
            mask = np.zeros(dist_map.shape, dtype=np.int32)
            
            # Process each contour
            for i, contour in enumerate(contours):
                # Skip invalid contours
                if contour is None or len(contour) < 3:
                    continue
                    
                try:
                    M = cv2.moments(contour)

                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        # Handle zero division - use contour center
                        cX = int(np.mean([p[0][0] for p in contour]))
                        cY = int(np.mean([p[0][1] for p in contour]))

                    # Set closest pixel to centroid
                    cY = min(max(0, cY), mask.shape[0]-1)
                    cX = min(max(0, cX), mask.shape[1]-1)
                    mask[cY, cX] = i + 1
                except Exception as e:
                    logger.error(f"Error processing contour {i}: {str(e)}")
                    continue

            # Use watershed to create final segmentation
            if np.max(mask) == 0:
                logger.warning("No centroids found - returning empty segmentation")
                return np.zeros(dist_map.shape, dtype=np.int32)
                
            labels = watershed(-dist_map, mask, mask=cell_fill).astype(np.int32)

            return labels
        except Exception as e:
            logger.error(f"Error in cells_from_dist_map: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(dist_map.shape, dtype=np.int32)

    def run(self, image, return_dist_map=False, cells_max=None, cell_fill=None):
        """Run the SAMCell pipeline on an input image
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
        return_dist_map : bool, optional
            Whether to return the distance map as well, by default False
        cells_max_threshold : float, optional
            Override default cells_max threshold
        cell_fill_threshold : float, optional
            Override default cell_fill threshold
            
        Returns
        -------
        numpy.ndarray
            Segmentation labels
        numpy.ndarray, optional
            Distance map if return_dist_map is True
        """
        try:
            # Make a copy to avoid modifying original
            image = image.copy()
            
            # Run prediction
            dist_map = self.predict_on_full_img(image)

            self.cells_max_threshold = cells_max if cells_max is not None else self.cells_max_threshold
            self.cell_fill_threshold = cell_fill if cell_fill is not None else self.cell_fill_threshold
            
            # Extract cells from distance map using provided or default thresholds
            labels = self.cells_from_dist_map(dist_map, self.cells_max_threshold, self.cell_fill_threshold)
            
            # Log results
            num_cells = len(np.unique(labels)) - 1  # -1 to exclude background
            logger.info(f"Segmentation complete. Found {num_cells} cells.")
            
            if return_dist_map:
                return labels, dist_map
            else:
                return labels
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            logger.error(traceback.format_exc())
            if return_dist_map:
                return np.zeros(image.shape[:2], dtype=np.int32), np.zeros(image.shape[:2], dtype=np.float32)
            else:
                return np.zeros(image.shape[:2], dtype=np.int32)

    def run_batch_thresholds(self, image, cells_max_values, cell_fill_values):
        """Run inference on a single image with multiple threshold values.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image
        cells_max_values : List[float]
            List of cells_max threshold values to try
        cell_fill_values : List[float]
            List of cell_fill threshold values to try
            
        Returns
        -------
        Dict[Tuple[float, float], numpy.ndarray]
            Dictionary mapping (cells_max, cell_fill) tuples to segmentation labels
        """
        try:
            # Make a copy to avoid modifying original
            image = image.copy()
            
            # Get the distance map once
            logger.info(f"Running batch threshold analysis with {len(cells_max_values)} x {len(cell_fill_values)} combinations")
            dist_map = self.predict_on_full_img(image)
            
            # Process all threshold combinations
            results = {}
            total_combinations = len(cells_max_values) * len(cell_fill_values)
            
            combo_index = 0
            for cells_max in cells_max_values:
                # Recompute contours and markers for each cells_max value
                cells_max_mask = dist_map > cells_max
                
                # Convert to binary mask
                cells_max_mask = cells_max_mask.astype(np.uint8)
                
                # Find contours - handle different OpenCV versions
                try:
                    contours, _ = cv2.findContours(cells_max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    _, contours, _ = cv2.findContours(cells_max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                logger.info(f"Found {len(contours)} contours with cells_max={cells_max}")
                
                # Create base mask with markers at centroids
                base_mask = np.zeros(dist_map.shape, dtype=np.int32)
                for i, contour in enumerate(contours):
                    if contour is None or len(contour) < 3:
                        continue
                        
                    try:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            # Handle zero division - use contour center
                            cX = int(np.mean([p[0][0] for p in contour]))
                            cY = int(np.mean([p[0][1] for p in contour]))
                            
                        # Set closest pixel to centroid
                        cY = min(max(0, cY), base_mask.shape[0]-1)
                        cX = min(max(0, cX), base_mask.shape[1]-1)
                        base_mask[cY, cX] = i + 1
                    except Exception as e:
                        logger.error(f"Error processing contour {i}: {str(e)}")
                        continue
                
                for cell_fill in cell_fill_values:
                    combo_index += 1
                    # Create cell fill mask for current threshold
                    cell_fill_mask = dist_map > cell_fill
                    cell_fill_mask = cell_fill_mask.astype(np.uint8)
                    
                    # Apply watershed with current thresholds
                    if np.max(base_mask) == 0:
                        logger.warning(f"No centroids found for cells_max={cells_max}, cell_fill={cell_fill}")
                        labels = np.zeros(dist_map.shape, dtype=np.int32)
                    else:
                        labels = watershed(-dist_map, base_mask, mask=cell_fill_mask).astype(np.int32)
                    
                    # Store results
                    results[(cells_max, cell_fill)] = labels
                    
                    # Log progress
                    num_cells = len(np.unique(labels)) - 1  # -1 to exclude background
                    logger.info(f"Combination cells_max={cells_max}, cell_fill={cell_fill}: found {num_cells} cells")
            
            return results
        except Exception as e:
            logger.error(f"Error in run_batch_thresholds: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def calculate_metrics(self, labels: np.ndarray, original_image: Optional[np.ndarray] = None, 
                         include_texture: bool = False) -> 'pd.DataFrame':
        """
        Calculate comprehensive metrics for segmented cells.
        
        Parameters
        ----------
        labels : np.ndarray
            Labeled segmentation mask
        original_image : np.ndarray, optional
            Original grayscale image for intensity-based metrics
        include_texture : bool
            Whether to include texture metrics (computationally expensive)
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing all metrics for each cell
        """
        return calculate_all_metrics(labels, original_image, include_texture)
    
    def export_metrics(self, labels: np.ndarray, output_path: str, 
                      original_image: Optional[np.ndarray] = None,
                      include_texture: bool = False, include_statistics: bool = True) -> bool:
        """
        Calculate and export metrics to CSV file.
        
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
        include_statistics : bool
            Whether to include metric statistics (mean, median, std, etc.)
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        return export_metrics_csv(labels, output_path, original_image, include_texture, 
                                 include_summary=True, include_statistics=include_statistics)
    
    def export_metrics_excel(self, labels: np.ndarray, output_path: str, 
                            original_image: Optional[np.ndarray] = None,
                            include_texture: bool = False) -> bool:
        """
        Calculate and export metrics to Excel file with separate sheets.
        
        Parameters
        ----------
        labels : np.ndarray
            Labeled segmentation mask
        output_path : str
            Path to save the Excel file (.xlsx)
        original_image : np.ndarray, optional
            Original grayscale image
        include_texture : bool
            Whether to include texture metrics
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        return export_metrics_excel(labels, output_path, original_image, include_texture)
    
    def get_metric_statistics(self, labels: np.ndarray, original_image: Optional[np.ndarray] = None,
                             include_texture: bool = False) -> 'pd.DataFrame':
        """
        Calculate summary statistics for all metrics.
        
        Parameters
        ----------
        labels : np.ndarray
            Labeled segmentation mask
        original_image : np.ndarray, optional
            Original grayscale image for intensity-based metrics
        include_texture : bool
            Whether to include texture metrics (computationally expensive)
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing statistics (mean, median, std, etc.) for each metric
        """
        df = calculate_all_metrics(labels, original_image, include_texture)
        return calculate_metric_statistics(df)
    
    def get_basic_metrics(self, labels: np.ndarray, original_image: Optional[np.ndarray] = None):
        """
        Get basic metrics compatible with the GUI.
        
        Parameters
        ----------
        labels : np.ndarray
            Labeled segmentation mask
        original_image : np.ndarray, optional
            Original grayscale image
            
        Returns
        -------
        tuple
            (cell_count, avg_cell_area, confluency_str, avg_neighbors)
        """
        return compute_gui_metrics(labels, original_image)