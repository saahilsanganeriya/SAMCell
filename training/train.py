"""
Comprehensive training script for SAMCell.

This script supports:
- Multiple dataset training (concatenated)
- Early stopping
- Mixed precision training
- Comprehensive wandb logging
- Model checkpointing
- Command-line arguments
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import sys
from pathlib import Path
from torch.utils.data import ConcatDataset
import tempfile

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

# Add parent directory to path to import samcell
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from samcell.model import FinetunedSAM
from samcell.utils import (
    lr_warmup, init_wandb, log_gradient_stats,
    log_predictions, log_training_metrics, log_epoch_stats
)
from samcell.dataset_livecell import SAMDataset
from transformers import SamProcessor

# Import mixed precision training
try:
    from torch.cuda.amp import autocast, GradScaler
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=7, min_delta=1e-4, minimum_epochs=35):
        """
        Parameters
        ----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum change in loss to qualify as an improvement
        minimum_epochs : int
            Minimum number of epochs before early stopping can trigger
        """
        self.patience = patience
        self.min_delta = min_delta
        self.minimum_epochs = minimum_epochs
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.current_epoch = 0

    def __call__(self, val_loss):
        """Check if training should stop.

        Parameters
        ----------
        val_loss : float
            Current validation/training loss

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        self.current_epoch += 1

        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.current_epoch >= self.minimum_epochs:
                self.early_stop = True
                return True
            return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SAMCell on multiple datasets')
    parser.add_argument('--datasets', nargs='+', required=True,
                      help='List of dataset paths. Each path should contain imgs.npy, dist_maps.npy, and optionally wms.npy')
    parser.add_argument('--sam-model', type=str, default='facebook/sam-vit-base',
                      choices=['facebook/sam-vit-base', 'facebook/sam-vit-large', 'facebook/sam-vit-huge'],
                      help='SAM model variant to use')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=40,
                      help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                      help='Weight decay for optimizer')
    parser.add_argument('--no-wandb', action='store_true',
                      help='Disable wandb logging')
    parser.add_argument('--patience', type=int, default=7,
                      help='Number of epochs to wait before early stopping')
    parser.add_argument('--min-delta', type=float, default=1e-4,
                      help='Minimum change in loss to be considered as improvement')
    parser.add_argument('--output-dir', type=str, default='../checkpoints',
                      help='Output directory for checkpoints')
    parser.add_argument('--finetune-vision', action='store_true',
                      help='Fine-tune vision encoder (default: True)')
    parser.add_argument('--no-finetune-vision', action='store_true',
                      help='Do not fine-tune vision encoder')
    parser.add_argument('--save-interval', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--pretrained-weights', type=str, default=None,
                      help='Path to pretrained SAMCell weights (.pt file) to start training from')
    return parser.parse_args()


def setup_datasets(dataset_paths, sam_model):
    """Setup and concatenate multiple datasets.

    Parameters
    ----------
    dataset_paths : list of str
        List of paths to datasets
    sam_model : str
        SAM model identifier

    Returns
    -------
    Dataset or ConcatDataset
        Combined dataset
    """
    datasets = []
    for path in dataset_paths:
        path = Path(path)
        img_path = path / 'imgs.npy'
        ann_path = path / 'dist_maps.npy'
        weight_path = path / 'wms.npy' if (path / 'wms.npy').exists() else None

        if not img_path.exists() or not ann_path.exists():
            raise FileNotFoundError(f"Dataset files not found in {path}")

        print(f'Loading dataset from {path}...')
        processor = SamProcessor.from_pretrained(sam_model)
        dataset = SAMDataset(str(img_path), str(ann_path), processor, str(weight_path) if weight_path else None)
        datasets.append(dataset)
        print(f'Loaded {len(dataset)} images from {path}')

    if len(datasets) > 1:
        print('Concatenating datasets...')
        combined_dataset = ConcatDataset(datasets)
        print(f'Total images after concatenation: {len(combined_dataset)}')
        return combined_dataset
    return datasets[0]


def get_output_path(dataset_paths, sam_model, output_dir):
    """Generate output path based on datasets and model.

    Parameters
    ----------
    dataset_paths : list of str
        List of dataset paths
    sam_model : str
        SAM model identifier
    output_dir : str
        Base output directory

    Returns
    -------
    str
        Output path for checkpoints
    """
    # Extract dataset names from paths
    dataset_names = [Path(path).name for path in dataset_paths]
    dataset_str = '+'.join(dataset_names)

    # Extract model variant name
    model_variant = sam_model.split('/')[-1]

    # Create output path
    output_path = Path(output_dir) / f'samcell-{dataset_str}-{model_variant}'
    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def get_run_name(dataset_paths, sam_model, batch_size, learning_rate):
    """Generate a descriptive name for the wandb run.

    Parameters
    ----------
    dataset_paths : list of str
        List of dataset paths
    sam_model : str
        SAM model identifier
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate

    Returns
    -------
    str
        Run name
    """
    # Extract dataset names from paths
    dataset_names = [Path(path).name for path in dataset_paths]
    dataset_str = '+'.join(dataset_names)

    # Extract model variant name
    model_variant = sam_model.split('/')[-1]

    # Create run name
    run_name = f"{dataset_str}-{model_variant}-bs{batch_size}-lr{learning_rate:.1e}"
    return run_name


def main():
    """Main training function."""
    args = parse_args()

    # Setup wandb
    do_log_wandb = not args.no_wandb and WANDB_AVAILABLE
    run = None
    if do_log_wandb:
        run_name = get_run_name(args.datasets, args.sam_model, args.batch_size, args.learning_rate)
        run = init_wandb(run_name, project="SAMCell", config={
            "datasets": args.datasets,
            "sam_model": args.sam_model,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "finetune_vision": not args.no_finetune_vision,
            "pretrained_weights": args.pretrained_weights,
        })

    # Setup dataset and dataloader
    dataset = setup_datasets(args.datasets, args.sam_model)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Setup model
    print(f'Loading model {args.sam_model}...')
    finetune_vision = not args.no_finetune_vision
    modelHelper = FinetunedSAM(args.sam_model, finetune_vision=finetune_vision, finetune_prompt=False, finetune_decoder=True)
    model = modelHelper.get_model()

    # Load pretrained weights if provided
    if args.pretrained_weights:
        print(f'Loading pretrained weights from {args.pretrained_weights}...')
        try:
            state_dict = torch.load(args.pretrained_weights, map_location='cpu')
            model.load_state_dict(state_dict, strict=True)
            print('✓ Successfully loaded pretrained weights')
        except Exception as e:
            print(f'Warning: Could not load pretrained weights: {e}')
            print('Starting from base SAM model instead...')

    # Setup training
    print('Setting up training...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_warmup)
    sigmoid = nn.Sigmoid()
    l2_loss = nn.MSELoss(reduction='mean')

    # Setup mixed precision training
    scaler = GradScaler() if (torch.cuda.is_available() and MIXED_PRECISION_AVAILABLE) else None

    # Get output path
    output_path = get_output_path(args.datasets, args.sam_model, args.output_dir)
    print(f'Model will be saved to {output_path}')

    # Training loop
    model.train()
    step = 0
    best_loss = float('inf')
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    for epoch in range(args.num_epochs):
        epoch_losses = []
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}'):
            # forward pass with mixed precision if CUDA is available
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                multimask_output=True)
                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                    step_loss = l2_loss(sigmoid(predicted_masks[:, 0]), ground_truth_masks)
            else:
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                            multimask_output=True)
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                step_loss = l2_loss(sigmoid(predicted_masks[:, 0]), ground_truth_masks)

            # backward pass with gradient scaling if CUDA is available
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(step_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                step_loss.backward()
                optimizer.step()

            scheduler.step()
            epoch_losses.append(step_loss.item())

            # Log metrics
            if do_log_wandb:
                current_lr = float(scheduler.get_last_lr()[0])
                log_training_metrics(step_loss, epoch, step, current_lr, model)
                log_predictions(batch, sigmoid(predicted_masks), ground_truth_masks, step)

            step += 1

        # Log epoch statistics
        mean_loss = np.mean(epoch_losses)
        if do_log_wandb:
            log_epoch_stats(epoch_losses, epoch)

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean_loss:.6f}')

        # Save checkpoint every N epochs
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(output_path, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')

            if do_log_wandb and run:
                # Save model state to wandb artifact
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                    torch.save(model.state_dict(), tmp.name)
                    artifact = wandb.Artifact(
                        f'model-checkpoint-epoch-{epoch}-run-{run.id}',
                        type='model',
                        description=f'Model checkpoint at epoch {epoch} from run {run.id}'
                    )
                    artifact.add_file(tmp.name, name='model.pt')
                    run.log_artifact(artifact)
                    os.unlink(tmp.name)  # Clean up temp file

        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_model_path = os.path.join(output_path, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model to {best_model_path} (loss: {best_loss:.6f})')

            if do_log_wandb and run:
                # Save best model to wandb artifact
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                    torch.save(model.state_dict(), tmp.name)
                    artifact = wandb.Artifact(
                        f'best-model-run-{run.id}',
                        type='model',
                        description=f'Best model with loss {best_loss:.4f} from run {run.id}'
                    )
                    artifact.add_file(tmp.name, name='model.pt')
                    run.log_artifact(artifact)
                    os.unlink(tmp.name)  # Clean up temp file

        # Early stopping check
        if early_stopping(mean_loss):
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

    # Save final model
    final_model_path = os.path.join(output_path, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model to {final_model_path}')

    if do_log_wandb and run:
        # Save final model to wandb artifact
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            torch.save(model.state_dict(), tmp.name)
            artifact = wandb.Artifact(
                f'final-model-run-{run.id}',
                type='model',
                description=f'Final model after training from run {run.id}'
            )
            artifact.add_file(tmp.name, name='model.pt')
            run.log_artifact(artifact)
            os.unlink(tmp.name)  # Clean up temp file

        wandb.finish()

    print('Training complete!')


if __name__ == '__main__':
    main()
