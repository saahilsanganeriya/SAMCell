import cv2
import math
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
import wandb

from transformers import SamModel, SamConfig, SamMaskDecoderConfig
from transformers.models.sam.modeling_sam import SamMaskDecoder, SamVisionConfig
from transformers.models.sam import convert_sam_original_to_hf_format

# LoRA qkv class from SAMed GitHub repo (https://github.com/hitachinsk/SAMed/blob/main/sam_lora_image_encoder.py)
class _LoRA_qv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        self.qkv.requires_grad_(False)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))

        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
    
class FinetunedSAM():
    '''a helper class to handle setting up SAM from the transformers library for finetuning
    '''
    def __init__(self, sam_model, finetune_vision=False, finetune_prompt=True, finetune_decoder=True, lora_vision=True, LoRA_rank=4):
        self.model = SamModel.from_pretrained(sam_model).to('cuda')
        self.w_As = []
        self.w_Bs = []
        self.LoRA_rank = LoRA_rank

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

        #add lora to vision encoder if needed.  Following SAMed, we only apply to the q and v weights
        if lora_vision:
            for t_layer_i, blk in enumerate(self.model.vision_encoder.layers):
                w_qkv_linear = blk.attn.qkv
                dim = w_qkv_linear.in_features

                w_qkv_linear = blk.attn.qkv
                # r = w_qkv_linear.out_features // self.LoRA_rank
                w_a_linear_q = nn.Linear(dim, self.LoRA_rank, bias=False)
                w_b_linear_q = nn.Linear(self.LoRA_rank, dim, bias=False)
                w_a_linear_v = nn.Linear(dim, self.LoRA_rank, bias=False)
                w_b_linear_v = nn.Linear(self.LoRA_rank, dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
                blk.attn.qkv = _LoRA_qv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            
            self.init_lora_weights()
        
    def init_lora_weights(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5)) #also, nn.init.normal_(self.lora_down.weight, std=1 / r)
            w_A.requires_grad_(True)
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            w_B.requires_grad_(True)

    def get_model(self):
        return self.model
    
    def load_weights(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path))