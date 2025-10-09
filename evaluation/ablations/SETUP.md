# Setting Up Ablation Scripts

The complete ablation study scripts from the paper experiments are available in `../../SAMCell-paper/src/`.

## Option 1: Copy Scripts from SAMCell-paper (Recommended)

If you want to run the exact ablation studies from the paper:

```bash
# Copy the main ablation script
cp ../../SAMCell-paper/src/ablation_studies_extended.py ./run_ablations.py

# Copy evaluation utilities if needed
cp ../../SAMCell-paper/src/evaluate_checkpoints.py ../evaluate_checkpoints.py

# Copy visualization scripts
cp ../../SAMCell-paper/ablations_vis/*.py ./
```

Then update the import paths in the copied files:
- Change `from model import FinetunedSAM` to `from samcell.model import FinetunedSAM`
- Change `from pipeline import SlidingWindowPipeline` to `from samcell.pipeline import SlidingWindowPipeline`
- Change `from utils import ...` to `from samcell.utils import ...`

## Option 2: Create New Ablation Scripts

If you want to create custom ablation studies, use the template structure below.

### Template: Basic Ablation Runner

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from samcell.model import FinetunedSAM
from samcell.pipeline import SlidingWindowPipeline
import pandas as pd
import numpy as np

def run_ablation_study(model_path, dataset_path, param_grid):
    """
    Run ablation study over parameter grid.

    Parameters
    ----------
    model_path : str
        Path to model weights
    dataset_path : str
        Path to dataset
    param_grid : dict
        Dictionary of parameters to test

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    # Load model
    model = FinetunedSAM('facebook/sam-vit-base')
    model.load_weights(model_path)

    # Load dataset
    imgs = np.load(f"{dataset_path}/imgs.npy")
    anns = np.load(f"{dataset_path}/anns.npy")

    results = []

    for param_name, param_values in param_grid.items():
        for value in param_values:
            # Create pipeline with parameter
            pipeline = SlidingWindowPipeline(
                model=model,
                device='cuda',
                **{param_name: value}
            )

            # Run inference
            predictions = [pipeline.run(img) for img in imgs]

            # Calculate metrics (implement your metric function)
            metric = calculate_metric(predictions, anns)

            results.append({
                'parameter': param_name,
                'value': value,
                'metric': metric
            })

    return pd.DataFrame(results)
```

## Available Scripts from SAMCell-paper

The SAMCell-paper repository contains the following evaluation/ablation scripts:

1. **ablation_studies_extended.py** (~1000 lines)
   - Comprehensive ablation runner
   - 5 different studies (patch size, datasets, models, pretraining, per-epoch)
   - Parallel processing with ThreadPoolExecutor
   - WandB integration
   - Automatic checkpointing and retry logic

2. **evaluate_checkpoints.py**
   - Batch evaluation of WandB checkpoints
   - Grid search over threshold parameters
   - CTC metric calculation

3. **evaluate_checkpoints_parallel.py**
   - Parallelized version of checkpoint evaluation

4. **Visualization scripts** (in ablations_vis/)
   - dataset_combos_ablation_vis.py
   - patch_size_ablation_vis.py
   - pretraining_ablation_vis.py
   - sam_models_ablation_vis.py
   - threshold_ablation_vis.py

## Dependencies for Full Ablation Suite

```bash
pip install stardist pandas matplotlib seaborn wandb psutil
```

## Cell Tracking Challenge Binaries

Download evaluation binaries from:
https://celltrackingchallenge.net/evaluation-methodology/

Place in: `SAMCell/evaluation/cell-tracking-binaries/`
