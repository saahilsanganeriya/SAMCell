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
# Load pretraining ablation data
print("Loading pretraining ablation data...")
pretraining_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell/src/ablations_final/study_4_pretraining.csv"

if os.path.exists(pretraining_file):
    df = pd.read_csv(pretraining_file)
    print(f"Loaded data from {pretraining_file} with {len(df)} entries")
else:
    print(f"File {pretraining_file} not found.")

#%%
# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Number of unique datasets: {df['dataset'].nunique()}")
print(f"Available datasets: {df['dataset'].unique()}")

# Extract model types based on 'random-weights' prefix
df['is_pretrained'] = ~df['run_name'].str.contains('random-weights')
df['model_label'] = df['is_pretrained'].map({True: 'Pretrained', False: 'Random Weights'})

print(f"Model types: {df['model_label'].unique()}")

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

# Function to create a visualization of pretraining impact on metrics
def create_pretraining_visualization(data, save_path=None):
    """Create a visualization showing the impact of pretraining on segmentation metrics."""
    
    # Prepare data
    metrics = ['seg_value', 'det_value', 'csb_value']
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the color palette
    unique_models = sorted(data['model_label'].unique())
    palette = sns.color_palette("Set2", n_colors=len(unique_models))
    
    # Create x-axis positions
    x = np.arange(len(metrics))
    width = 0.35  # Width of bars
    
    # Plot bars for each model
    for i, model in enumerate(unique_models):
        values = [data[data['model_label'] == model][metric].mean() for metric in metrics]
        bars = ax.bar(x + (i - 0.5) * width, values, width, 
                     label=model,
                     color=palette[i],
                     alpha=0.8)
        
        # Add value annotations
        for j, v in enumerate(values):
            ax.text(x[j] + (i - 0.5) * width, v + 0.02, 
                   f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Impact of Pretraining on Performance\n(Averaged across PBL_HEK and PBL_N2A)', 
                 fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_display_names[m] for m in metrics], fontsize=12)
    
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#%%
# Create pretraining visualization for the filtered datasets
save_dir = "paper/pretraining_ablation"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "pretraining_ablation.png")

# Create and save the figure
create_pretraining_visualization(df_filtered, save_path)

#%%
# Dataset-specific visualization
# Create a separate plot for each dataset to see how pretraining affects different datasets

# Set up a figure with one subplot per dataset
datasets = df_filtered['dataset'].unique()
fig, axes = plt.subplots(1, len(datasets), figsize=(18, 8), sharey=True)

for i, dataset in enumerate(datasets):
    dataset_data = df_filtered[df_filtered['dataset'] == dataset]
    
    # Set the color palette
    unique_models = sorted(dataset_data['model_label'].unique())
    palette = sns.color_palette("Set2", n_colors=len(unique_models))
    
    # Create x-axis positions
    metrics = ['seg_value', 'det_value', 'csb_value']
    x = np.arange(len(metrics))
    width = 0.35  # Width of bars
    
    # Plot bars for each model
    for j, model in enumerate(unique_models):
        model_data = dataset_data[dataset_data['model_label'] == model]
        values = [model_data[metric].mean() for metric in metrics]
        bars = axes[i].bar(x + (j - 0.5) * width, values, width, 
                         label=model,
                         color=palette[j],
                         alpha=0.8)
        
        # Add value annotations
        for k, v in enumerate(values):
            axes[i].text(x[k] + (j - 0.5) * width, v + 0.02, 
                       f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the subplot
    axes[i].set_title(f'{dataset}', fontsize=14)
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([metric_display_names[m] for m in metrics], fontsize=12)
    axes[i].set_ylim(0, 1.0)
    axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Only add y-label to the first subplot
    if i == 0:
        axes[i].set_ylabel('Score', fontsize=14)
    
    # Only add legend to the last subplot
    if i == len(datasets) - 1:
        axes[i].legend(loc='upper right', fontsize=12)

plt.suptitle('Impact of Pretraining on Different Datasets', fontsize=16, y=1.05)
plt.tight_layout()

# Save the figure
dataset_save_path = os.path.join(save_dir, "pretraining_ablation_by_dataset.png")
plt.savefig(dataset_save_path, dpi=300, bbox_inches='tight')
print(f"Dataset-specific figure saved to {dataset_save_path}")

#%%
# Summary analysis
print("\n" + "="*80)
print("PRETRAINING ABLATION STUDY SUMMARY")
print("="*80)

# Calculate average performance for each model type
for model in sorted(df_filtered['model_label'].unique()):
    model_data = df_filtered[df_filtered['model_label'] == model]
    
    print(f"\nModel: {model}")
    print("-" * 40)
    
    # Print average metrics
    for metric in ['seg_value', 'det_value', 'csb_value']:
        avg_value = model_data[metric].mean()
        print(f"Average {metric_display_names[metric]}: {avg_value:.4f}")
    
    # Print per-dataset performance
    for dataset in df_filtered['dataset'].unique():
        dataset_data = model_data[model_data['dataset'] == dataset]
        if len(dataset_data) > 0:
            print(f"\n  {dataset}:")
            for metric in ['seg_value', 'det_value', 'csb_value']:
                value = dataset_data[metric].iloc[0]
                print(f"    {metric_display_names[metric]}: {value:.4f}")

# Calculate performance improvement from pretraining
print("\n" + "-" * 40)
print("Performance Improvement from Pretraining:")

pretrained_data = df_filtered[df_filtered['model_label'] == 'Pretrained']
random_data = df_filtered[df_filtered['model_label'] == 'Random Weights']

for metric in ['seg_value', 'det_value', 'csb_value']:
    pretrained_avg = pretrained_data[metric].mean()
    random_avg = random_data[metric].mean()
    improvement = pretrained_avg - random_avg
    improvement_pct = (improvement / random_avg) * 100
    
    print(f"{metric_display_names[metric]} improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

# Find overall best model
best_model = df_filtered.loc[df_filtered['csb_value'].idxmax()]

print("\n" + "-" * 40)
print("Best Model Configuration:")
print(f"Model: {best_model['model_label']}")
print(f"Dataset: {best_model['dataset']}")
print(f"OP_CSB Score: {best_model['csb_value']:.4f}")
print("="*80) 