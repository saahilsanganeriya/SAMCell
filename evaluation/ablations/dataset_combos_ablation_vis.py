#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib.ticker as ticker

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math

#%%
# Load dataset combinations ablation data
print("Loading dataset combinations ablation data...")
dataset_combos_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell/src/ablations_final/study_2_dataset_combos.csv"

if os.path.exists(dataset_combos_file):
    df = pd.read_csv(dataset_combos_file)
    print(f"Loaded data from {dataset_combos_file} with {len(df)} entries")
else:
    print(f"File {dataset_combos_file} not found.")

#%%
# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Number of unique datasets: {df['dataset'].nunique()}")
print(f"Available datasets: {df['dataset'].unique()}")
print(f"Training configurations: {sorted(df['run_name'].unique())}")

# Show the first few rows
df.head()

#%%
# Filter out CellPose-test dataset
df_filtered = df[df['dataset'] != 'CellPose-test']
print(f"Filtered dataset shape: {df_filtered.shape}")
print(f"Remaining datasets: {df_filtered['dataset'].unique()}")

#%%
# Define metric display names
metric_display_names = {
    'seg_value': 'SEG',
    'det_value': 'DET',
    'csb_value': r'$\mathrm{OP}_{\mathrm{CSB}}$'
}

# Function to create a visualization of dataset combinations vs. metrics
def create_dataset_combos_visualization(data, save_path=None):
    """Create a visualization showing the impact of dataset combinations on segmentation metrics."""
    
    # Prepare data
    metrics = ['seg_value', 'det_value', 'csb_value']
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the color palette
    unique_configs = sorted(data['run_name'].unique())
    palette = sns.color_palette("husl", n_colors=len(unique_configs))
    
    # Create x-axis positions
    x = np.arange(len(metrics))
    width = 0.25  # Width of bars
    
    # Create readable config labels
    config_labels = [
        config.replace('CellPose-train+LiveCell-train-sam-vit-base-bs8-lr1.0e-04', 'CellPose+LiveCell')
             .replace('CellPose-train-sam-vit-base-bs8-lr1.0e-04', 'CellPose')
             .replace('LiveCell-train-sam-vit-base-bs4-lr1.0e-04', 'LiveCell')
        for config in unique_configs
    ]
    
    # Plot bars for each configuration
    for i, (config, label) in enumerate(zip(unique_configs, config_labels)):
        values = [data[data['run_name'] == config][metric].mean() for metric in metrics]
        bars = ax.bar(x + (i - 1) * width, values, width, 
                     label=label,
                     color=palette[i],
                     alpha=0.8)
        
        # Add value annotations
        for j, v in enumerate(values):
            ax.text(x[j] + (i - 1) * width, v + 0.02, 
                   f'{v:.3f}', ha='center', va='bottom', fontsize=16)
    
    # Customize the plot
    ax.set_ylabel('Score', fontsize=28)
    #ax.set_title('Impact of Training Dataset Combinations on Performance\n(Averaged across PBL_HEK and PBL_N2A)', fontsize=32, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_display_names[m] for m in metrics], fontsize=24)
    
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=20)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#%%
# Create dataset combinations visualization for the filtered datasets
save_dir = "paper"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "dataset_combos_ablation.png")

# Create and save the figure
create_dataset_combos_visualization(df_filtered, save_path)

#%%
# Summary analysis
print("\n" + "="*80)
print("DATASET COMBINATIONS ABLATION STUDY SUMMARY")
print("="*80)

# Calculate average performance for each training configuration
for config in sorted(df_filtered['run_name'].unique()):
    config_data = df_filtered[df_filtered['run_name'] == config]
    
    print(f"\nTraining Configuration: {config}")
    print("-" * 40)
    
    # Print average metrics
    for metric in ['seg_value', 'det_value', 'csb_value']:
        avg_value = config_data[metric].mean()
        print(f"Average {metric_display_names[metric]}: {avg_value:.4f}")
    
    # Print per-dataset performance
    for dataset in df_filtered['dataset'].unique():
        dataset_data = config_data[config_data['dataset'] == dataset]
        if len(dataset_data) > 0:
            print(f"\n  {dataset}:")
            for metric in ['seg_value', 'det_value', 'csb_value']:
                value = dataset_data[metric].iloc[0]
                print(f"    {metric_display_names[metric]}: {value:.4f}")

# Find best configuration based on average CSB
config_performance = {}
for config in df_filtered['run_name'].unique():
    config_data = df_filtered[df_filtered['run_name'] == config]
    config_performance[config] = config_data['csb_value'].mean()

best_config = max(config_performance.items(), key=lambda x: x[1])

print("\n" + "-" * 40)
print("Best Training Configuration:")
print(f"Configuration: {best_config[0]}")
print(f"Average OP_CSB: {best_config[1]:.4f}")
print("="*80) 