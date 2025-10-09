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
# Load SAM models ablation data
print("Loading SAM models ablation data...")
sam_models_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell/src/ablations_final/study_3_sam_models.csv"

if os.path.exists(sam_models_file):
    df = pd.read_csv(sam_models_file)
    print(f"Loaded data from {sam_models_file} with {len(df)} entries")
else:
    print(f"File {sam_models_file} not found.")

#%%
# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Number of unique datasets: {df['dataset'].nunique()}")
print(f"Available datasets: {df['dataset'].unique()}")
print(f"Model types: {df['model_type'].unique()}")

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

# Function to create a visualization of SAM models vs. metrics
def create_sam_models_visualization(data, save_path=None):
    """Create a visualization showing the impact of different SAM models on segmentation metrics."""
    
    # Prepare data
    metrics = ['seg_value', 'det_value', 'csb_value']
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the color palette
    unique_models = sorted(data['model_type'].unique())
    palette = sns.color_palette("husl", n_colors=len(unique_models))
    
    # Create x-axis positions
    x = np.arange(len(metrics))
    width = 0.25  # Width of bars
    
    # Create readable model labels
    model_labels = [
        model.replace('facebook/sam-vit-base', 'SAM-ViT-B')
             .replace('facebook/sam-vit-large', 'SAM-ViT-L')
        for model in unique_models
    ]
    
    # Plot bars for each model
    for i, (model, label) in enumerate(zip(unique_models, model_labels)):
        values = [data[data['model_type'] == model][metric].mean() for metric in metrics]
        bars = ax.bar(x + (i - 1) * width, values, width, 
                     label=label,
                     color=palette[i],
                     alpha=0.8)
        
        # Add value annotations
        for j, v in enumerate(values):
            ax.text(x[j] + (i - 1) * width, v + 0.02, 
                   f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Impact of SAM Model Size on Performance\n(Averaged across PBL_HEK and PBL_N2A)', 
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
# Create SAM models visualization for the filtered datasets
save_dir = "paper/sam_models_ablation"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "sam_models_ablation.png")

# Create and save the figure
create_sam_models_visualization(df_filtered, save_path)

#%%
# Summary analysis
print("\n" + "="*80)
print("SAM MODELS ABLATION STUDY SUMMARY")
print("="*80)

# Calculate average performance for each model
for model in sorted(df_filtered['model_type'].unique()):
    model_data = df_filtered[df_filtered['model_type'] == model]
    
    print(f"\nModel: {model}")
    print("-" * 40)
    
    # Print average metrics
    for metric in ['seg_value', 'det_value', 'csb_value']:
        avg_value = model_data[metric].mean()
        print(f"Average {metric_display_names[metric]}: {avg_value:.6f}")
    
    # Print per-dataset performance
    for dataset in df_filtered['dataset'].unique():
        dataset_data = model_data[model_data['dataset'] == dataset]
        if len(dataset_data) > 0:
            print(f"\n  {dataset}:")
            for metric in ['seg_value', 'det_value', 'csb_value']:
                value = dataset_data[metric].iloc[0]
                print(f"    {metric_display_names[metric]}: {value:.6f}")

# Find best model based on average CSB
model_performance = {}
for model in df_filtered['model_type'].unique():
    model_data = df_filtered[df_filtered['model_type'] == model]
    model_performance[model] = model_data['csb_value'].mean()

best_model = max(model_performance.items(), key=lambda x: x[1])

print("\n" + "-" * 40)
print("Best Model:")
print(f"Model: {best_model[0]}")
print(f"Average OP_CSB: {best_model[1]:.6f}")
print("="*80) 