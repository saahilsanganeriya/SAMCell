from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import cv2
from torchvision import transforms
import torch
import random
import time

class SAMDataset(Dataset):
    def __init__(self, img_path, ann_path, processor, weight_path=None, crop_size=256):
        self.imgs = np.load(img_path)
        self.anns = np.load(ann_path)
        if weight_path is not None:
            self.weights = np.load(weight_path)
        else:
            self.weights = None

        self.processor = processor
        self.crop_size = crop_size

        self.isTrain = 'train' in img_path or 'train' in ann_path

    def __len__(self):
        return len(self.imgs)
    
    def _preprocess(self, img):

        #convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #adaptive hist normalization
        # img_norm = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8)).apply(img)

        #grab 2nd derivative via laplacian
        # edges = cv2.Laplacian(img_norm, cv2.CV_64F, ksize=3)
        # edges = img_norm

        #grab 1st derivative via sobel
        # edges = cv2.Sobel(img_norm, cv2.CV_64F, 1, 1, ksize=3)
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        #cvt to 3 channel
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        return img
    
    def _data_augmentation(self, img, label, weight):
        #random flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)
            if weight is not None:
                weight = cv2.flip(weight, 1)

        #random rotate + scale
        angle = random.randint(-180, 180)
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, scale)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        label = cv2.warpAffine(label, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
        if weight is not None:
            weight = cv2.warpAffine(weight, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

        #prevent overflow
        img = img.astype(np.float32)

        #random brightness
        alpha = random.uniform(0.95, 1.05)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)

        # #random contrast / gamma
        # beta = random.randint(-1, 1)
        # gamma = random.uniform(0.9, 1.1)
        # img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)
        # img = np.power(img, gamma)

        #back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        #randomly invert image
        if random.random() > 0.5:
            img = cv2.bitwise_not(img)

        return img, label, weight

    def __getitem__(self, idx):
        img = self.imgs[idx]
        image_orig = img.copy()
        label = self.anns[idx]
        if self.weights is not None:
            weight = self.weights[idx]
        else:
            weight = None

        #preprocess image
        img = self._preprocess(img)

        if self.isTrain:
            #data augmentation
            img, label, weight = self._data_augmentation(img, label, weight)

        #random crop to size
        x = random.randint(0, img.shape[1] - self.crop_size)
        y = random.randint(0, img.shape[0] - self.crop_size)
        img = img[y:y+self.crop_size, x:x+self.crop_size]
        label = label[y:y+self.crop_size, x:x+self.crop_size]
        if self.weights is not None:
            weight = weight[y:y+self.crop_size, x:x+self.crop_size]

        # return image_orig, img, label, weight #just for debugging

        # prepare image and prompt for the model
        inputs = self.processor(img, return_tensors="pt") #input image: shape: (256, 256, 3), range [0, 255]

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        inputs['ground_truth_mask'] = label
        # inputs['binary_mask'] = label > 0.001

        if self.weights is not None:
            inputs['weight'] = torch.tensor(weight).float()

        return inputs