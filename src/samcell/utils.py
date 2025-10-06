"""
Utility functions for training SAMCell models.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


def init_wandb(run_name=None, project="SAMCell", config=None):
    """Initialize wandb with optional run name and configuration.

    Parameters
    ----------
    run_name : str, optional
        Name for the wandb run
    project : str, optional
        Wandb project name (default: "SAMCell")
    config : dict, optional
        Configuration dictionary to log to wandb

    Returns
    -------
    run : wandb.Run or None
        Wandb run object if successful, None otherwise
    """
    if not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Logging disabled.")
        return None

    try:
        if run_name:
            run = wandb.init(project=project, name=run_name)
        else:
            run = wandb.init(project=project)

        if config is not None:
            wandb.config.update(config)

        return run
    except Exception as e:
        print(f"Warning: wandb initialization failed: {e}")
        return None


def log_wandb(run, current_step, learning_rate, loss):
    """Log basic metrics to wandb.

    Parameters
    ----------
    run : wandb.Run or None
        Wandb run object
    current_step : int
        Current training step
    learning_rate : float
        Current learning rate
    loss : float
        Current loss value
    """
    if run is not None:
        try:
            run.log({"lr": learning_rate, "loss": loss}, step=current_step)
        except:
            pass  # Fail silently if wandb is not available


def log_gradient_stats(model, step):
    """Log gradient statistics to wandb.

    Parameters
    ----------
    model : torch.nn.Module
        Model to log gradients from
    step : int
        Current training step
    """
    if not WANDB_AVAILABLE or step % 100 != 0:
        return

    try:
        grad_norms = []
        grad_means = []
        grad_stds = []

        for p in model.parameters():
            if p.grad is not None:
                grad_norms.append(torch.norm(p.grad).item())
                grad_means.append(p.grad.mean().item())
                grad_stds.append(p.grad.std().item())

        if grad_norms:
            wandb.log({
                "gradients/norm": np.mean(grad_norms),
                "gradients/mean": np.mean(grad_means),
                "gradients/std": np.mean(grad_stds)
            }, step=step)
    except Exception as e:
        print(f"Warning: gradient logging failed: {e}")


def log_predictions(batch, predicted_masks, ground_truth_masks, step):
    """Log prediction visualizations to wandb.

    Parameters
    ----------
    batch : dict
        Batch dictionary containing pixel_values
    predicted_masks : torch.Tensor
        Predicted masks
    ground_truth_masks : torch.Tensor
        Ground truth masks
    step : int
        Current training step
    """
    if not WANDB_AVAILABLE or step % 100 != 0:
        return

    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Ensure pixel values are in correct range for display
        img_vis = batch["pixel_values"][0].permute(1, 2, 0).cpu().numpy()
        img_vis = ((img_vis - img_vis.min()) / (img_vis.max() - img_vis.min()) * 255).astype(np.uint8)

        axes[0, 0].imshow(img_vis)
        axes[0, 0].set_title("Input Image")
        axes[0, 1].imshow(ground_truth_masks[0].cpu().numpy(), cmap='jet')
        axes[0, 1].set_title("Ground Truth")
        axes[1, 0].imshow(predicted_masks[0, 0].detach().cpu().numpy(), cmap='jet')
        axes[1, 0].set_title("Prediction")
        axes[1, 1].imshow(np.abs(predicted_masks[0, 0].detach().cpu().numpy() - ground_truth_masks[0].cpu().numpy()), cmap='jet')
        axes[1, 1].set_title("Error")

        plt.tight_layout()
        wandb.log({"predictions": wandb.Image(plt)}, step=step)
        plt.close()
    except Exception as e:
        print(f"Warning: prediction logging failed: {e}")


def log_system_metrics():
    """Log system resource usage metrics.

    Returns
    -------
    metrics : dict
        Dictionary containing system metrics
    """
    metrics = {}

    if PSUTIL_AVAILABLE:
        try:
            metrics.update({
                "system/cpu_percent": psutil.cpu_percent(),
                "system/memory_percent": psutil.virtual_memory().percent
            })
        except:
            pass

    if torch.cuda.is_available():
        try:
            metrics.update({
                "system/memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "system/memory_reserved_mb": torch.cuda.memory_reserved() / 1024**2
            })
        except:
            pass

    return metrics


def log_training_metrics(step_loss, epoch, step, learning_rate, model):
    """Log comprehensive training metrics to wandb.

    Parameters
    ----------
    step_loss : torch.Tensor or float
        Current step loss
    epoch : int
        Current epoch
    step : int
        Current training step
    learning_rate : float
        Current learning rate
    model : torch.nn.Module
        Model being trained
    """
    if not WANDB_AVAILABLE:
        return

    try:
        param_norm = torch.norm(torch.stack([torch.norm(p) for p in model.parameters()])).item()

        metrics = {
            "train/step_loss": step_loss.item() if torch.is_tensor(step_loss) else step_loss,
            "train/epoch": epoch,
            "train/step": step,
            "train/learning_rate": learning_rate,
            "model/param_norm": param_norm,
        }

        # Add system metrics
        metrics.update(log_system_metrics())

        wandb.log(metrics, step=step)
    except Exception as e:
        print(f"Warning: training metrics logging failed: {e}")


def log_epoch_stats(epoch_losses, epoch=None):
    """Log epoch-level statistics to wandb.

    Parameters
    ----------
    epoch_losses : list
        List of losses from the epoch
    epoch : int, optional
        Epoch number
    """
    if not WANDB_AVAILABLE or not epoch_losses:
        return

    try:
        stats = {
            "epoch/mean_loss": np.mean(epoch_losses),
            "epoch/std_loss": np.std(epoch_losses),
            "epoch/min_loss": np.min(epoch_losses),
            "epoch/max_loss": np.max(epoch_losses)
        }

        if epoch is not None:
            stats["epoch"] = epoch
            stats["epoch/loss"] = np.mean(epoch_losses)

        wandb.log(stats)
    except Exception as e:
        print(f"Warning: epoch stats logging failed: {e}")


def lr_warmup(current_step, warmup_steps=500, decay_rate=0.9999):
    """Learning rate warmup and decay schedule.

    Parameters
    ----------
    current_step : int
        Current training step
    warmup_steps : int, optional
        Number of warmup steps (default: 500)
    decay_rate : float, optional
        Decay rate after warmup (default: 0.9999)

    Returns
    -------
    float
        Learning rate multiplier
    """
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step / warmup_steps)
    else:
        # Very gentle exponential decay after warmup
        decay_steps = current_step - warmup_steps
        return decay_rate ** decay_steps
