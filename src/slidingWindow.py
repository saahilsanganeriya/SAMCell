import cv2
import numpy as np

class SlidingWindowHelper:
    def __init__(self, crop_size: int, overlap_size: int):
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        # Create a blending mask that decreases towards the edges
        self._create_blending_mask()
    
    def _create_blending_mask(self):
        """Create a blending mask that decreases towards the edges for smooth transitions."""
        mask = np.ones((self.crop_size, self.crop_size), dtype=np.float32)
        
        # Calculate a more gradual falloff for edges 
        # Use a cosine function for smoother transition
        for i in range(self.overlap_size):
            # Calculate weight factor (increases as we move from edge to center)
            # Using cosine falloff for smoother transition
            weight = 0.5 * (1 - np.cos(np.pi * i / self.overlap_size))
            
            # Apply to all four edges
            mask[i, :] *= weight  # Top edge
            mask[-i-1, :] *= weight  # Bottom edge
            mask[:, i] *= weight  # Left edge
            mask[:, -i-1] *= weight  # Right edge
        
        self.blending_mask = mask

    def seperate_into_crops(self, img):
        # If image is smaller than crop size, adjust the crop size
        orig_height, orig_width = img.shape
        
        # For debugging
        # print(f"Original image size: {orig_height}x{orig_width}")
        
        if orig_height < self.crop_size or orig_width < self.crop_size:
            # Set crop size to the smaller dimension of the image
            new_crop_size = min(orig_height, orig_width)
            if new_crop_size < 2 * self.overlap_size:
                # If even the smaller dimension is too small for proper overlap
                # Just process the whole image at once
                new_overlap_size = 0
                # Return the entire image as a single crop
                return [img], [(0, 0, orig_width, orig_height)], (0, 0, orig_width, orig_height)
            else:
                # Adjust crop size and keep the original overlap
                new_overlap_size = self.overlap_size
        else:
            # Use the original crop size and overlap
            new_crop_size = self.crop_size
            new_overlap_size = self.overlap_size
        
        # Calculate the effective stride (distance between consecutive crop centers)
        stride = new_crop_size - 2 * new_overlap_size
        
        # Mirror the image such that the edges are repeated around the overlap region
        img_mirrored = cv2.copyMakeBorder(img, new_overlap_size, new_overlap_size, new_overlap_size, new_overlap_size, cv2.BORDER_REFLECT)

        # Get the image dimensions after mirroring
        mirrored_height, mirrored_width = img_mirrored.shape
        
        # Calculate exact number of crops needed to cover the entire image
        # Add 1 because we need at least one crop even if the image is smaller than the stride
        num_crops_y = max(1, int(np.ceil(orig_height / stride)))
        num_crops_x = max(1, int(np.ceil(orig_width / stride)))
        
        # For debugging
        # print(f"Using crop size: {new_crop_size}, overlap: {new_overlap_size}, stride: {stride}")
        # print(f"Number of crops: {num_crops_y}x{num_crops_x}")
        
        # Initialize a list to store cropped images
        cropped_images = []
        orig_regions = []
        crop_unique_region = (new_overlap_size, new_overlap_size, new_crop_size - 2 * new_overlap_size, new_crop_size - 2 * new_overlap_size)
        
        # Create an array to visualize crop coverage (for debugging)
        # debug_coverage = np.zeros((orig_height, orig_width), dtype=np.float32)
        
        for y_idx in range(num_crops_y):
            for x_idx in range(num_crops_x):
                # For all but the last tile in each dimension, use regular spacing
                if y_idx < num_crops_y - 1:
                    y_start = y_idx * stride + new_overlap_size
                else:
                    # Last row - ensure it aligns with the bottom edge
                    y_start = mirrored_height - new_crop_size
                
                if x_idx < num_crops_x - 1:
                    x_start = x_idx * stride + new_overlap_size
                else:
                    # Last column - ensure it aligns with the right edge
                    x_start = mirrored_width - new_crop_size
                
                # Calculate crop boundaries in mirrored image
                x_end = x_start + new_crop_size
                y_end = y_start + new_crop_size
                
                # Extract the crop with mirrored edges
                crop = img_mirrored[y_start:y_end, x_start:x_end]
                
                # Calculate the corresponding region in the original image
                # The -new_overlap_size is to account for the mirroring we did earlier
                orig_x = max(0, x_start - new_overlap_size)
                orig_y = max(0, y_start - new_overlap_size)
                
                # Make sure we don't exceed the original image size
                orig_w = min(new_crop_size, orig_width - orig_x)
                orig_h = min(new_crop_size, orig_height - orig_y)
                
                # Ensure we're not trying to go outside the original image
                orig_x = min(orig_x, orig_width)
                orig_y = min(orig_y, orig_height)
                
                # Fix potential negative width/height
                orig_w = max(0, min(orig_w, orig_width - orig_x))
                orig_h = max(0, min(orig_h, orig_height - orig_y))
                
                orig_region = (orig_x, orig_y, orig_w, orig_h)
                
                # For debugging - visualize coverage
                # if orig_w > 0 and orig_h > 0:
                #     debug_coverage[orig_y:orig_y+orig_h, orig_x:orig_x+orig_w] += 1
                
                # Don't add empty regions
                if orig_w > 0 and orig_h > 0:
                    # Append the cropped image and region
                    cropped_images.append(crop)
                    orig_regions.append(orig_region)
        
        # Debug - check coverage (should all be >= 1)
        # coverage_gaps = np.where(debug_coverage == 0)
        # if coverage_gaps[0].size > 0:
        #     print(f"WARNING: Found {coverage_gaps[0].size} uncovered pixels")
        
        # Debug - check overlap (should be around 2-4 in overlap areas)
        # heavy_overlap = np.where(debug_coverage > 4)
        # if heavy_overlap[0].size > 0:
        #     print(f"WARNING: Found {heavy_overlap[0].size} pixels with excessive overlap (>4)")
        
        return cropped_images, orig_regions, crop_unique_region
    
    def separate_into_crops(self, img):
        """
        Correctly spelled version of seperate_into_crops.
        """
        return self.seperate_into_crops(img)

    def combine_crops(self, orig_size, cropped_images, orig_regions, crop_unique_region, sam_outputs=None):
        """
        Combine crops back into a full image with smooth blending.
        
        Args:
            orig_size: Original image size (H, W)
            cropped_images: List of image crops
            orig_regions: List of regions in the original image
            crop_unique_region: Region in each crop considered unique
            sam_outputs: Optional SAM model outputs to use instead of cropped_images
                        (SAM outputs are typically 256x256 regardless of input crop size)
            dist_maps: Alternative name for sam_outputs (for backward compatibility)
        
        Returns:
            Combined image of orig_size
        """
        # For backward compatibility, if dist_maps is provided and sam_outputs is not,
        # use dist_maps as sam_outputs
        if sam_outputs is None and dist_maps is not None:
            sam_outputs = dist_maps
            
        # Initialize output image and weight accumulator for blending
        output_img = np.zeros(orig_size, dtype=np.float32)
        weight_map = np.zeros(orig_size, dtype=np.float32)
        
        # Handle potential empty crops list
        if not cropped_images:
            return output_img
        
        # Get the actual crop size from the first crop
        crop_h, crop_w = cropped_images[0].shape
        
        # Create a blending mask for this specific crop size if needed
        if (crop_h != self.crop_size or crop_w != self.crop_size):
            temp_crop_size = self.crop_size
            self.crop_size = max(crop_h, crop_w)
            self._create_blending_mask()
            blending_mask = self.blending_mask
            self.crop_size = temp_crop_size
        else:
            blending_mask = self.blending_mask

        # Ensure blending mask matches crop size
        if blending_mask.shape != (crop_h, crop_w):
            blending_mask = cv2.resize(blending_mask, (crop_w, crop_h))
        
        # For debugging, create a visualization of which regions are being covered
        # debug_coverage = np.zeros(orig_size, dtype=np.float32)
            
        for i, (crop, region) in enumerate(zip(cropped_images, orig_regions)):
            # Extract region coordinates
            x, y, w, h = region
            
            # Skip if the region is invalid
            if w <= 0 or h <= 0:
                continue
            
            # Record coverage for debugging
            # debug_coverage[y:y+h, x:x+w] += 1
            
            # If we have SAM outputs, use them instead of the crop
            if sam_outputs is not None:
                sam_output = sam_outputs[i]
                # Resize SAM output to match crop size if needed
                if sam_output.shape != crop.shape:
                    sam_output = cv2.resize(sam_output, (crop.shape[1], crop.shape[0]), 
                                          interpolation=cv2.INTER_LINEAR)
                crop_to_use = sam_output
            else:
                crop_to_use = crop
            
            # Calculate which part of the blending mask to use
            # This ensures we only use the portion of the mask that corresponds
            # to the valid part of the crop
            mask_h, mask_w = min(h, blending_mask.shape[0]), min(w, blending_mask.shape[1])
            
            # Ensure we're not exceeding the dimensions of the output image
            y_end = min(y + mask_h, orig_size[0])
            x_end = min(x + mask_w, orig_size[1])
            mask_h = y_end - y
            mask_w = x_end - x
            
            if mask_h <= 0 or mask_w <= 0:
                continue
            
            # Get the region of the crop and mask to use
            crop_region = crop_to_use[:mask_h, :mask_w]
            mask_region = blending_mask[:mask_h, :mask_w]
            
            # Apply the blending mask to this region of the crop
            weighted_crop = crop_region * mask_region
            
            # Add to the output image and weight map at the correct region
            output_img[y:y_end, x:x_end] += weighted_crop
            weight_map[y:y_end, x:x_end] += mask_region
        
        # Check for any gaps in the weight map (should be no zeros)
        # zero_weights = np.where(weight_map == 0)
        # if zero_weights[0].size > 0:
        #     print(f"WARNING: Found {zero_weights[0].size} pixels with zero weight")
        #     print(f"Positions: y={zero_weights[0][:10]}, x={zero_weights[1][:10]}")
        
        # Avoid division by zero and normalize
        mask = weight_map > 0.0001
        output_img[mask] /= weight_map[mask]
        
        # Handle any remaining zeros by filling with nearest neighbor
        if np.any(~mask):
            # Create a mask for where we have values
            valid_mask = np.zeros(orig_size, dtype=np.uint8)
            valid_mask[mask] = 1
            
            # Use distance transform to find nearest valid pixel
            dist, indices = cv2.distanceTransformWithLabels(
                1 - valid_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            
            # Get coordinates of nearest valid pixels
            h, w = orig_size
            coords_y, coords_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            nearest_y = coords_y.flatten()[indices.flatten() - 1].reshape(h, w)
            nearest_x = coords_x.flatten()[indices.flatten() - 1].reshape(h, w)
            
            # Fill in missing values with nearest valid pixel
            for y in range(h):
                for x in range(w):
                    if not mask[y, x]:
                        ny, nx = nearest_y[y, x], nearest_x[y, x]
                        if 0 <= ny < h and 0 <= nx < w:
                            output_img[y, x] = output_img[ny, nx]
        
        return output_img