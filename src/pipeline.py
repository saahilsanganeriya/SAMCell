from typing import Any
from skimage.segmentation import watershed
import math
import torch
from torch import nn
import numpy as np
import cv2
from transformers import SamProcessor
from slidingWindow import SlidingWindowHelper

class SlidingWindowPipeline:
    def __init__(self, model, device, crop_size=256):
        self.model = model.get_model()
        self.device = device
        self.crop_size = crop_size
        self.sigmoid = nn.Sigmoid()
        self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
        self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)

    def _preprocess(self, img):
        # img = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)

        #grab 2nd derivative via laplacian
        # edges = cv2.Laplacian(img_norm, cv2.CV_64F, ksize=3)

        #grab 1st derivative via sobel
        # edges = cv2.Sobel(img_norm, cv2.CV_64F, 1, 1, ksize=3)
        
        # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #convert to color if necessary
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        #default SAM preprocessing
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
            outputs_finetuned = self.model(pixel_values=image,  multimask_output=True)

        prob_finetuned = outputs_finetuned['pred_masks'].squeeze(1)

        #sigmoid
        dist_map = self.sigmoid(prob_finetuned)[0][0]

        return dist_map

    def spilt_into_crops(self, image_orig):
        crops = []
        #split into 512x512 crops
        for i in range(0, math.ceil(image_orig.shape[0] / (self.crop_size)) + 1):
            for j in range(0, math.ceil(image_orig.shape[1] / self.crop_size) + 1):
                min_x = i * self.crop_size
                min_y = j * self.crop_size
                min_x = min(min_x, image_orig.shape[0] - self.crop_size)
                min_y = min(min_y, image_orig.shape[1] - self.crop_size)
                crops.append((image_orig[min_x:min_x+self.crop_size, min_y:min_y+self.crop_size], (min_x, min_y)))

        return crops

    def predict_on_full_img(self, image_orig):
        orig_shape = image_orig.shape
        #rescale to 1000 (biggest dimension)
        if image_orig.shape[0] > image_orig.shape[1]:
            new_size = (int(image_orig.shape[1] * (1000 / image_orig.shape[0])), 1000)
        else:
            new_size = (1000, int(image_orig.shape[0] * (1000 / image_orig.shape[1])))

        # image_orig = cv2.resize(image_orig, (new_size))

        # crops = self.spilt_into_crops(image_orig)
        crops, orig_regions, crop_unique_region = self.sliding_window_helper.seperate_into_crops(image_orig)

        #predict on crops
        dist_maps = []
        for crop in crops:
            dist_map = self.get_model_prediction(crop).cpu().numpy()

            #resize to crop size
            dist_map = cv2.resize(dist_map, (self.crop_size, self.crop_size))
            dist_maps.append(dist_map)

        #reconstruct image
        # cell_dist_map = np.zeros(image_orig.shape)
        # for crop in crops:
        #     min_x,min_y = crop[1]
        #     cell_dist_map[min_x:min_x+self.crop_size, min_y:min_y+self.crop_size] = dist_maps.pop(0)

        cell_dist_map = self.sliding_window_helper.combine_crops(orig_shape, dist_maps, orig_regions, crop_unique_region)

        return cell_dist_map

    def cells_from_dist_map(self, dist_map):
        cells_max = dist_map > 0.5
        cell_fill = dist_map > 0.05
        #find centroids of connected components
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

            #set closest pixel to centroid
            mask[int(cY), int(cX)] = i + 1


        labels = watershed(-dist_map, mask, mask=cell_fill).astype(np.int32)

        return labels
    
    def run(self, image, return_dist_map=False):
        dist_map = self.predict_on_full_img(image)
        labels = self.cells_from_dist_map(dist_map)

        if return_dist_map:
            return labels, dist_map
        
        return labels