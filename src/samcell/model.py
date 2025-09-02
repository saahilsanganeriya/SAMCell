import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np

from transformers import SamModel, SamConfig, SamMaskDecoderConfig
from transformers.models.sam.modeling_sam import SamMaskDecoder, SamVisionConfig
from safetensors.torch import load_file
import itertools
import numpy as np
import logging
import traceback
import gc
import os
from pathlib import Path

class FinetunedSAM():
    '''a helper class to handle setting up SAM from the transformers library for finetuning
    '''
    def __init__(self, sam_model, finetune_vision=False, finetune_prompt=True, finetune_decoder=True):
        self.model = SamModel.from_pretrained(sam_model)
        #freeze required layers
        if not finetune_vision:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad_(False)
        else:
            for param in self.model.vision_encoder.parameters():
                param.requires_grad_(True)
            
        if not finetune_prompt:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad_(False)
        
        if not finetune_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad_(False)
        else:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad_(True)

    def get_model(self):
        return self.model
    
    def load_weights(self, weight_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")):
        """Load weights from a file to the SAM model.
        
        Supports multiple file formats including .pt, .bin, and .safetensors.
        
        Parameters
        ----------
        weight_path : str
            Path to the weights file
        map_location : str or torch.device, optional
            Device to load the weights to, by default None
            
        Raises
        ------
        Exception
            If weights cannot be loaded
        """

        try:
            # Determine file format based on extension
            file_ext = Path(weight_path).suffix.lower()
            
            if file_ext == '.safetensors':
                # Use safetensors if available
                try:
                    import safetensors.torch
                    state_dict = safetensors.torch.load_file(weight_path)
                except ImportError:
                    state_dict = torch.load(weight_path, map_location=map_location)
            else:
                # For .pt, .bin, and any other format, use PyTorch's load
                state_dict = torch.load(weight_path, map_location=map_location)
            
            # Load the state dictionary
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise

    def load_weights_safetensors(self, weight_path):
        self.model.load_state_dict(load_file(weight_path))

    def load_weights_pt(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")))