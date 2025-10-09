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
# Load patch size ablation data
print("Loading patch size ablation data...")
patch_size_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell/src/ablations_final/study_1_patch_sizes.csv"

if os.path.exists(patch_size_file):
    df = pd.read_csv(patch_size_file)
    print(f"Loaded data from {patch_size_file} with {len(df)} entries")
else:
    print(f"File {patch_size_file} not found.")

#%%
# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Number of unique datasets: {df['dataset'].nunique()}")
print(f"Available datasets: {df['dataset'].unique()}")
print(f"Crop sizes: {sorted(df['crop_size'].unique())}")

# Show the first few rows
df.head()

#%%
# Filter out CellPose-test dataset
df_filtered = df[df['dataset'] != 'CellPose-test']
print(f"Filtered dataset shape: {df_filtered.shape}")
print(f"Remaining datasets: {df_filtered['dataset'].unique()}")

#%%
# Function to create a visualization of patch size vs. metrics
def create_patch_size_visualization(data, save_path=None):
    """Create a visualization showing the impact of patch size on segmentation metrics."""
    
    # Prepare data
    metrics = ['seg_value', 'det_value', 'csb_value']
    metric_display_names = {
        'seg_value': 'SEG',
        'det_value': 'DET',
        'csb_value': r'$\mathrm{OP}_{\mathrm{CSB}}$'
    }
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set the color palette
    palette = sns.color_palette("tab10", n_colors=len(data['dataset'].unique()))
    
    # Plot for each dataset and metric
    for i, dataset in enumerate(sorted(data['dataset'].unique())):
        dataset_data = data[data['dataset'] == dataset]
        
        # Sort by crop size
        dataset_data = dataset_data.sort_values('crop_size')
        
        # Create line plots for each metric
        for j, metric in enumerate(metrics):
            # Use different line styles for each metric
            linestyle = ['-', '--', '-.'][j]
            
            # Plot the line
            ax.plot(
                dataset_data['crop_size'], 
                dataset_data[metric], 
                marker='o',
                linestyle=linestyle,
                linewidth=2,
                color=palette[i],
                label=f"{dataset} - {metric_display_names[metric]}"
            )
            
            # Add value annotations
            for x, y in zip(dataset_data['crop_size'], dataset_data[metric]):
                ax.annotate(
                    f"{y:.6f}", 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
    
    # Set up the plot 
    ax.set_xlabel('Patch Size', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Impact of Patch Size on Segmentation Performance', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(sorted(data['crop_size'].unique()))
    ax.set_ylim(0, 1.0)
    
    # Add legend
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#%%
# Create patch size visualization for the filtered datasets
save_dir = "paper/patch_size_ablation"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "patch_size_ablation.png")

# Create and save the figure
create_patch_size_visualization(df_filtered, save_path)

#%%
# Create visualizations for individual datasets
for dataset in df_filtered['dataset'].unique():
    dataset_data = df_filtered[df_filtered['dataset'] == dataset]
    save_dataset_path = os.path.join(save_dir, f"patch_size_ablation_{dataset}.png")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare metrics
    metrics = ['seg_value', 'det_value', 'csb_value']
    metric_display_names = {
        'seg_value': 'SEG',
        'det_value': 'DET',
        'csb_value': r'$\mathrm{OP}_{\mathrm{CSB}}$'
    }
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    
    # Sort by crop size
    dataset_data = dataset_data.sort_values('crop_size')
    
    # Create line plots for each metric
    for j, metric in enumerate(metrics):
        ax.plot(
            dataset_data['crop_size'], 
            dataset_data[metric], 
            marker='o',
            linestyle='-',
            linewidth=2,
            color=colors[j],
            label=f"{metric_display_names[metric]}"
        )
        
        # Add value annotations
        for x, y in zip(dataset_data['crop_size'], dataset_data[metric]):
            ax.annotate(
                f"{y:.6f}", 
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
            )
    
    # Set up the plot 
    ax.set_xlabel('Patch Size', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(f'Impact of Patch Size on {dataset} Segmentation Performance', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(sorted(dataset_data['crop_size'].unique()))
    ax.set_ylim(0, 1.0)
    
    # Add legend
    ax.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_dataset_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_dataset_path}")

#%%
# Create a bar chart visualization (alternative view)
save_bar_path = os.path.join(save_dir, "patch_size_ablation_bar.png")

fig, ax = plt.subplots(figsize=(14, 8))

# Set width of bars
bar_width = 0.2
index = np.arange(len(df_filtered['crop_size'].unique()))

# Plot bars for each dataset and metric
datasets = sorted(df_filtered['dataset'].unique())
metrics = ['seg_value', 'det_value', 'csb_value']
colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

for i, dataset in enumerate(datasets):
    dataset_data = df_filtered[df_filtered['dataset'] == dataset].sort_values('crop_size')
    
    # Group bars for each patch size
    for j, metric in enumerate(metrics):
        offset = (i * len(metrics) + j) * bar_width - (len(datasets) * len(metrics) * bar_width / 2) + bar_width/2
        
        bars = ax.bar(
            index + offset, 
            dataset_data[metric], 
            bar_width, 
            label=f"{dataset} - {metric_display_names[metric]}",
            color=colors[i],
            alpha=0.7 + j*0.1  # Varying alpha for different metrics
        )
        
        # Add value annotations
        for k, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.01,
                f"{height:.6f}",
                ha='center', 
                va='bottom',
                rotation=90,
                fontsize=8
            )

# Set up the axis and labels
patch_sizes = sorted(df_filtered['crop_size'].unique())
ax.set_xticks(index)
ax.set_xticklabels(patch_sizes)
ax.set_xlabel('Patch Size', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.set_title('Impact of Patch Size on Segmentation Performance by Dataset', fontsize=16)
ax.set_ylim(0, 1.0)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Add legend
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(datasets)*len(metrics)//2, fontsize=10)

plt.tight_layout()
plt.savefig(save_bar_path, dpi=300, bbox_inches='tight')
print(f"Bar chart saved to {save_bar_path}")

#%%
# Summary analysis
print("\n" + "="*80)
print("PATCH SIZE ABLATION STUDY SUMMARY")
print("="*80)

# Find best patch size for each dataset and metric
for dataset in df_filtered['dataset'].unique():
    dataset_data = df_filtered[df_filtered['dataset'] == dataset]
    
    print(f"\nDataset: {dataset}")
    print("-" * 40)
    
    for metric in ['seg_value', 'det_value', 'csb_value']:
        best_idx = dataset_data[metric].idxmax()
        best_row = dataset_data.loc[best_idx]
        
        print(f"Best {metric_display_names[metric]}: {best_row[metric]:.6f} at patch size {int(best_row['crop_size'])}")
    
    # Find best overall patch size (using CSB)
    best_csb_idx = dataset_data['csb_value'].idxmax()
    best_csb_row = dataset_data.loc[best_csb_idx]
    
    print(f"\nOptimal patch size for {dataset}: {int(best_csb_row['crop_size'])}")
    print(f"  SEG: {best_csb_row['seg_value']:.6f}")
    print(f"  DET: {best_csb_row['det_value']:.6f}")
    print(f"  OP_CSB: {best_csb_row['csb_value']:.6f}")

# Overall best patch size across all datasets (average CSB)
patch_size_avg = {}
for size in df_filtered['crop_size'].unique():
    size_data = df_filtered[df_filtered['crop_size'] == size]
    patch_size_avg[size] = size_data['csb_value'].mean()

best_avg_size = max(patch_size_avg.items(), key=lambda x: x[1])

print("\n" + "-" * 40)
print(f"Best overall patch size (averaged across datasets): {int(best_avg_size[0])}")
print(f"Average OP_CSB: {best_avg_size[1]:.6f}")
print("="*80) 