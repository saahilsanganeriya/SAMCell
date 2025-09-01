from typing import Any, Optional
from skimage.segmentation import watershed
import math
import torch
from torch import nn
import numpy as np
import cv2
from transformers import SamProcessor
from slidingWindow import SlidingWindowHelper
from skimage import measure
from tqdm import tqdm
from metrics import calculate_all_metrics, export_metrics_csv, compute_gui_metrics


class SlidingWindowPipeline:
    def __init__(self, model, device, crop_size=256, cells_max=0.5, cell_fill=0.05):
        self.model = model.get_model()
        self.device = device
        self.crop_size = crop_size
        self.sigmoid = nn.Sigmoid()
        self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
        self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)
        self.cells_max = cells_max
        self.cell_fill = cell_fill

    def _preprocess(self, img):
        # img = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)

        # grab 2nd derivative via laplacian
        # edges = cv2.Laplacian(img_norm, cv2.CV_64F, ksize=3)

        # grab 1st derivative via sobel
        # edges = cv2.Sobel(img_norm, cv2.CV_64F, 1, 1, ksize=3)

        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # convert to color if necessary
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # default SAM preprocessing
        inputs = self.processor(img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def _preprocess_sam(self, img):
        inputs = self.processor(img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def get_model_prediction(self, image):
        image_orig = image.copy()
        image = self._preprocess(image_orig)
        self.model.eval().to(self.device)

        # forward pass
        with torch.no_grad():
            outputs_finetuned = self.model(pixel_values=image, multimask_output=True)

        prob_finetuned = outputs_finetuned['pred_masks'].squeeze(1)

        # sigmoid
        dist_map = self.sigmoid(prob_finetuned)[0][0]

        return dist_map

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

    def cells_from_dist_map(self, dist_map):
        cells_max = dist_map > self.cells_max
        cell_fill = dist_map > self.cell_fill
        # find centroids of connected components
        contours, _ = cv2.findContours(cells_max.astype(np.uint8), 0, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(dist_map.shape, dtype=np.int32)
        # for i, contour in enumerate(contours):
        #     contour = np.flip(contour, axis=2)
        #     mask[tuple(contour.T)] = i + 1

        for i, contour in enumerate(contours):
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # Handle cases where the moment is zero to avoid division by zero
                cX, cY = 0, 0

            # set closest pixel to centroid
            mask[int(cY), int(cX)] = i + 1

        labels = watershed(-dist_map, mask, mask=cell_fill).astype(np.int32)

        return labels

    def run(self, image, return_dist_map=False, cells_max=None, cell_fill=None):
        """Run inference on a single image.
        Args:
            image: Input image
            return_dist_map: Whether to return the distance map
            cells_max: Override default cells_max threshold (optional)
            cell_fill: Override default cell_fill threshold (optional)
        """
        # Use instance defaults if not provided
        cells_max = cells_max if cells_max is not None else self.cells_max
        cell_fill = cell_fill if cell_fill is not None else self.cell_fill

        dist_map = self.predict_on_full_img(image)
        labels = self.cells_from_dist_map(dist_map)

        if return_dist_map:
            return labels, dist_map

        return labels

    def run_batch_thresholds(self, image, cells_max_values, cell_fill_values):
        """Run inference on a single image with multiple threshold values.
        Args:
            image: Input image
            cells_max_values: List of cells_max threshold values to try
            cell_fill_values: List of cell_fill threshold values to try
        Returns:
            Dictionary mapping (cells_max, cell_fill) tuples to segmentation labels
        """
        # Get the distance map once
        dist_map = self.predict_on_full_img(image)

        # Process all threshold combinations
        results = {}
        total_combinations = len(cells_max_values) * len(cell_fill_values)
        for cells_max in cells_max_values:
            # Recompute contours and markers for each cells_max value
            cells_max_mask = dist_map > cells_max
            contours, _ = cv2.findContours(cells_max_mask.astype(np.uint8), 0, cv2.CHAIN_APPROX_SIMPLE)
            base_mask = np.zeros(dist_map.shape, dtype=np.int32)

            for i, contour in enumerate(contours):
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                base_mask[int(cY), int(cX)] = i + 1

            for cell_fill in cell_fill_values:
                # Create cell fill mask for current threshold
                cell_fill_mask = dist_map > cell_fill

                # Apply watershed with current thresholds
                labels = watershed(-dist_map, base_mask, mask=cell_fill_mask).astype(np.int32)
                results[(cells_max, cell_fill)] = labels
        return results
    
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
                      include_texture: bool = False) -> bool:
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
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        return export_metrics_csv(labels, output_path, original_image, include_texture)
    
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