import numpy as np
import matplotlib.pyplot as plt
import monai
from tqdm import tqdm
import wandb

from transformers import SamProcessor, SamImageProcessor
import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

from model import FinetunedSAM
from utils import lr_warmup, init_wandb, log_wandb
from dataset_livecell import SAMDataset

dataset_path = '../datasets/CellPose-train/'
img_path = dataset_path + 'imgs.npy'
ann_path = dataset_path + 'dist_maps.npy'
weight_path = dataset_path + 'wms.npy'

sam_model = 'facebook/sam-vit-base'
output_path = '../checkpoints/samcell-cyto-lora'
num_epochs = 40
do_log_wandb = False

#setup custom dataset
print('loading dataset...')
processor = SamProcessor.from_pretrained(sam_model)
dataset = SAMDataset(img_path, ann_path, processor, weight_path=None)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print('loaded {} images'.format(len(dataset)))

print('loading model...')
loraHelper = FinetunedSAM(sam_model, LoRA_rank=4, finetune_vision=True, finetune_prompt=False, finetune_decoder=True, lora_vision=True)
model = loraHelper.get_model()

print('training...')
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.999))
scheduler = LambdaLR(optimizer, lr_lambda=lr_warmup)
sigmoid = nn.Sigmoid()
l2_loss = nn.MSELoss(reduction='mean')

if do_log_wandb:
    run = init_wandb()
   
model.train()
step = 0
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      multimask_output=True)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      step_loss = l2_loss(sigmoid(predicted_masks[:, 0]), ground_truth_masks)

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      step_loss.backward()

      # optimize
      optimizer.step()
      scheduler.step()
      epoch_losses.append(step_loss.item())

      #log (if needed)
      if do_log_wandb:
        log_wandb(run, step, float(scheduler.get_last_lr()[0]), step_loss.item())

      step += 1

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {np.mean(epoch_losses)}')

print('done training. saving model to {}...'.format(output_path))
model.save_pretrained(output_path)