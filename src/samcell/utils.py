def init_wandb():
    """Initialize wandb logging (requires wandb to be installed separately)."""
    try:
        import wandb
        run = wandb.init(project="cellseg")
        return run
    except ImportError:
        print("Warning: wandb not installed. Logging disabled.")
        return None
    
def log_wandb(run, current_step, learning_rate, loss):
    """Log metrics to wandb (requires wandb to be installed separately)."""
    if run is not None:
        try:
            run.log({"lr": learning_rate, "loss": loss}, step=current_step)
        except:
            pass  # Fail silently if wandb is not available

def lr_warmup(current_step):
    warmup_steps = 250
    training_steps = 10800 #65000 for livecell or 10800 for cellpose dataset
    if current_step < warmup_steps:  # current_step / warmup_steps * base_lr
        return float(current_step / warmup_steps)
    else:                                 # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))
