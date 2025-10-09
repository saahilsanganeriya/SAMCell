#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math

#%%
# Load threshold ablation data
print("Loading threshold ablation data...")
threshold_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell-paper/src/ablations_final/ablation_thresholds_final_old.csv"

if os.path.exists(threshold_file):
    df = pd.read_csv(threshold_file)
    print(f"Loaded data from {threshold_file} with {len(df)} entries")
else:
    print(f"File {threshold_file} not found.")
    # Create some sample data for testing if file does not exist
    import random
    data = []
    for dataset in ["PBL_HEK", "PBL_N2A", "CellPose-test"]:
        for cells_max in [round(x, 2) for x in np.arange(0.3, 0.56, 0.01)]:  # 0.01 increments
            for cell_fill in [round(x, 2) for x in np.arange(0, 0.21, 0.01)]:  # 0.01 increments
                seg = random.uniform(0.5, 0.9)
                det = random.uniform(0.5, 0.9)
                csb = (seg + det) / 2
                data.append({
                    "dataset": dataset,
                    "cells_max": cells_max,
                    "cell_fill": cell_fill,
                    "crop_size": 256,
                    "seg_value": seg,
                    "det_value": det,
                    "csb_value": csb,
                })
    df = pd.DataFrame(data)

#%%
# Filter out CellPose-test dataset
print("Filtering out CellPose-test dataset...")
df = df[df['dataset'] != 'CellPose-test']
print(f"After filtering, {len(df)} entries remain")

#%%
# Display basic info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Number of unique datasets: {df['dataset'].nunique()}")
print(f"Available datasets: {df['dataset'].unique()}")
if 'crop_size' in df.columns:
    print(f"Crop sizes: {sorted(df['crop_size'].unique())}")
print(f"cells_max values: {sorted(df['cells_max'].unique())}")
print(f"cell_fill values: {sorted(df['cell_fill'].unique())}")

# Show the first few rows
df.head()

#%%
# Data preprocessing
# Fill any missing values in metrics with 0
for metric in ['seg_value', 'det_value', 'csb_value']:
    if metric in df.columns:
        df[metric] = df[metric].fillna(0)

# Convert threshold values to numeric if they're not already
df['cells_max'] = pd.to_numeric(df['cells_max'])
df['cell_fill'] = pd.to_numeric(df['cell_fill'])

# No rounding to keep the full precision
# df['cells_max_rounded'] = df['cells_max'].round(2)
# df['cell_fill_rounded'] = df['cell_fill'].round(2)

# Rename columns for plotting (for easier reference)
metric_display_names = {
    'seg_value': 'SEG',
    'det_value': 'DET',
    'csb_value': r'$\mathrm{OP}_{\mathrm{CSB}}$'
}

#%%
# Function to create detailed metric heatmaps for a dataset
def create_detailed_metric_heatmaps(data, dataset_name="All Datasets", fig_size=(18, 6)):
    """Create detailed heatmaps for SEG, DET, and OP_CSB metrics."""
    fig, axes = plt.subplots(1, 3, figsize=fig_size, sharex=True, sharey=True)
    
    metrics = ['seg_value', 'det_value', 'csb_value']
    metric_titles = ['SEG Score', 'DET Score', r'$\mathrm{OP}_{\mathrm{CSB}}$ Score']
    
    # Custom colormap (white to blue)
    cmap = LinearSegmentedColormap.from_list('blue_gradient', ['#ffffff', '#1f77b4'])
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Create pivot table for this metric
        pivot = data.pivot_table(
            index='cells_max', 
            columns='cell_fill',
            values=metric,
            aggfunc='mean'
        )
        
        # Plot heatmap
        sns.heatmap(pivot, ax=axes[i], cmap=cmap, vmin=0, vmax=1, 
                   annot=False, linewidths=0, cbar=True)
        
        # Set title for this subplot
        axes[i].set_title(title, fontsize=14)
        
        # Add axis labels
        axes[i].set_xlabel('cell_fill threshold', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('cells_max threshold', fontsize=12)
        
        # Find and mark the best value for this metric
        best_idx = data[metric].idxmax()
        best_cells_max = data.loc[best_idx, 'cells_max']
        best_cell_fill = data.loc[best_idx, 'cell_fill']
        best_value = data.loc[best_idx, metric]
        
        # Get the position in the heatmap
        cells_max_values = sorted(data['cells_max'].unique())
        cell_fill_values = sorted(data['cell_fill'].unique())
        row_idx = cells_max_values.index(best_cells_max)
        col_idx = cell_fill_values.index(best_cell_fill)
        
        # Mark the best value
        axes[i].plot(col_idx + 0.5, row_idx + 0.5, 'o', color='red', markersize=10, markeredgecolor='white')
        axes[i].text(col_idx + 0.5, row_idx + 0.5, f'{best_value:.6f}', 
                    ha='center', va='center', color='white', fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.2'))
    
    plt.suptitle(f'Segmentation Metrics by Threshold Values - {dataset_name}', fontsize=16)
    plt.tight_layout()
    return fig

#%%
# 1. DETAILED HEATMAPS FOR EACH DATASET
# Create detailed heatmaps for each dataset individually

for dataset in df['dataset'].unique():
    # Filter data for this dataset
    dataset_data = df[df['dataset'] == dataset]
    
    # Create detailed metric heatmaps
    create_detailed_metric_heatmaps(dataset_data, dataset)

#%%
# 2. DETAILED HEATMAPS FOR ALL DATASETS COMBINED
# Create combined heatmaps averaging across all datasets

# Create detailed metric heatmaps for combined data
create_detailed_metric_heatmaps(df, "All Datasets", (18, 6))

#%%
# 3. IMPACT OF THRESHOLDS ON DIFFERENT METRICS
# Create line plots showing how each threshold affects metrics when the other is fixed

# First, analyze cells_max impact (fixing cell_fill at its median)
for dataset in df['dataset'].unique():
    # Get median cell_fill value
    median_cell_fill = df[df['dataset'] == dataset]['cell_fill'].median()
    closest_cell_fill = df.iloc[(df[df['dataset'] == dataset]['cell_fill'] - median_cell_fill).abs().argsort()[:1]]['cell_fill'].values[0]
    
    # Filter data
    fixed_cell_fill = df[(df['dataset'] == dataset) & (np.isclose(df['cell_fill'], closest_cell_fill))]
    
    plt.figure(figsize=(12, 6))
    
    # Sort by cells_max for line plot
    plot_data = fixed_cell_fill.sort_values('cells_max')
    
    # Plot metrics
    plt.plot(plot_data['cells_max'], plot_data['seg_value'], 'o-', label='SEG')
    plt.plot(plot_data['cells_max'], plot_data['det_value'], 's-', label='DET')
    plt.plot(plot_data['cells_max'], plot_data['csb_value'], '^-', label=r'$\mathrm{OP}_{\mathrm{CSB}}$')
    
    plt.title(f'Impact of cells_max (fixed cell_fill={closest_cell_fill:.6f}) - {dataset}')
    plt.xlabel('cells_max threshold')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

# Now analyze cell_fill impact (fixing cells_max at its median)
for dataset in df['dataset'].unique():
    # Get median cells_max value
    median_cells_max = df[df['dataset'] == dataset]['cells_max'].median()
    closest_cells_max = df.iloc[(df[df['dataset'] == dataset]['cells_max'] - median_cells_max).abs().argsort()[:1]]['cells_max'].values[0]
    
    # Filter data
    fixed_cells_max = df[(df['dataset'] == dataset) & (np.isclose(df['cells_max'], closest_cells_max))]
    
    plt.figure(figsize=(12, 6))
    
    # Sort by cell_fill for line plot
    plot_data = fixed_cells_max.sort_values('cell_fill')
    
    # Plot metrics
    plt.plot(plot_data['cell_fill'], plot_data['seg_value'], 'o-', label='SEG')
    plt.plot(plot_data['cell_fill'], plot_data['det_value'], 's-', label='DET')
    plt.plot(plot_data['cell_fill'], plot_data['csb_value'], '^-', label=r'$\mathrm{OP}_{\mathrm{CSB}}$')
    
    plt.title(f'Impact of cell_fill (fixed cells_max={closest_cells_max:.6f}) - {dataset}')
    plt.xlabel('cell_fill threshold')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

#%%
# 4. OPTIMAL THRESHOLD VISUALIZATION
# Create a 3D scatter plot to visualize the optimal threshold region

from mpl_toolkits.mplot3d import Axes3D

# Filter out cell_fill = 0.0 from the data before visualization
df_filtered = df[df['cell_fill'] > 0.0]

for dataset in df_filtered['dataset'].unique():
    # Filter data for this dataset
    subset = df_filtered[df_filtered['dataset'] == dataset]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Find min and max of csb_value for this dataset to set z-axis limits
    min_csb = subset['csb_value'].min()
    max_csb = subset['csb_value'].max()
    
    # Create scatter plot with OP_CSB value determining point size and color
    scatter = ax.scatter(
        subset['cells_max'], 
        subset['cell_fill'], 
        subset['csb_value'],
        c=subset['csb_value'], 
        s=subset['csb_value']*300,  # Further increased point size
        cmap='viridis',
        alpha=0.6
    )
    
    # Remove colorbar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label(r'$\mathrm{OP}_{\mathrm{CSB}}$ Score', fontsize=20)
    # cbar.ax.tick_params(labelsize=16)
    
    # Find and highlight the best point
    best_idx = subset['csb_value'].idxmax()
    best_point = subset.loc[best_idx]
    
    ax.scatter(
        best_point['cells_max'],
        best_point['cell_fill'],
        best_point['csb_value'],
        color='red',
        s=600,  # Further increased size of best point marker
        marker='*',
        zorder=100  # Ensure it's on top
    )
    
    # Add text box for best point with ALL CAPS
    best_text = f"BEST POINT\nCSB: {best_point['csb_value']:.3f}\nCELLS_MAX: {best_point['cells_max']:.3f}\nCELL_FILL: {best_point['cell_fill']:.3f}"
    
    ax.text(
        best_point['cells_max'],
        best_point['cell_fill'],
        best_point['csb_value'] + 0.02,
        best_text,
        color='black',
        fontweight='bold',
        fontsize=18,
        ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
    )
    
    # Configure 3D plot
    ax.set_xlabel('cells_max threshold', fontsize=26, labelpad=20, weight='bold')
    ax.set_ylabel('cell_fill threshold', fontsize=26, labelpad=20, weight='bold')
    ax.set_zlabel(r'$\mathrm{OP}_{\mathrm{CSB}}$ Score', fontsize=26, labelpad=20, weight='bold')
    
    # Set z-axis range to data min-max instead of 0-1
    margin = (max_csb - min_csb) * 0.1  # Add 10% margin
    ax.set_zlim(min_csb - margin, max_csb + margin)
    
    # Further reduce number of ticks on all axes
    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(3))
    
    # Increase tick label size
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    
    ax.set_title(f'Threshold Parameter Space - {dataset}', fontsize=30)
    
    # Make pane faces more transparent to improve visibility
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Make grid lines heavier and more visible
    ax.grid(True, linestyle='-', linewidth=1.5, alpha=0.5)
    
    # Adjust view angle - rotate HEK dataset 90 degrees anticlockwise
    if dataset == "PBL_HEK":
        ax.view_init(elev=20, azim=45)  # Rotate 90 degrees anticlockwise (-35 + 90 = 55)
    else:
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()

# Create a combined 3D plot with all datasets
plt.figure(figsize=(16, 14))
ax = plt.axes(projection='3d')

# Use different colors for each dataset
colors = plt.cm.tab10.colors
markers = ['o', 's', '^']

# Find global min and max across all datasets for consistent z-axis
global_min_csb = df_filtered['csb_value'].min()
global_max_csb = df_filtered['csb_value'].max()

for i, dataset in enumerate(df_filtered['dataset'].unique()):
    # Filter data for this dataset
    subset = df_filtered[df_filtered['dataset'] == dataset]
    
    # Create scatter plot with a different color for each dataset
    scatter = ax.scatter(
        subset['cells_max'], 
        subset['cell_fill'], 
        subset['csb_value'],
        c=[colors[i]]*len(subset),
        s=subset['csb_value']*200,  # Increased point size
        marker=markers[i % len(markers)],
        alpha=0.7,
        label=dataset
    )
    
    # Find and highlight the best point for this dataset
    best_idx = subset['csb_value'].idxmax()
    best_point = subset.loc[best_idx]
    
    ax.scatter(
        best_point['cells_max'],
        best_point['cell_fill'],
        best_point['csb_value'],
        color='red',
        s=600,  # Further increased size
        marker='*',
        edgecolors='black',
        linewidth=2,
        zorder=100  # Ensure it's on top
    )
    
    # Add annotation for this best point with larger font and box
    best_text = f"{dataset}\nBEST: {best_point['csb_value']:.3f}\nMAX: {best_point['cells_max']:.3f}\nFILL: {best_point['cell_fill']:.3f}"
    
    ax.text(
        best_point['cells_max'],
        best_point['cell_fill'],
        best_point['csb_value'] + 0.02,
        best_text,
        color='black',
        fontweight='bold',
        fontsize=18,
        ha='center',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5')
    )

# Configure combined 3D plot
ax.set_xlabel('cells_max threshold', fontsize=30, labelpad=25, weight='bold')
ax.set_ylabel('cell_fill threshold', fontsize=30, labelpad=25, weight='bold')
ax.set_zlabel(r'$\mathrm{OP}_{\mathrm{CSB}}$ Score', fontsize=30, labelpad=25, weight='bold')

# Set z-axis to data min-max with margin
margin = (global_max_csb - global_min_csb) * 0.1  # Add 10% margin
ax.set_zlim(global_min_csb - margin, global_max_csb + margin)

# Further reduce number of ticks on all axes
ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
ax.zaxis.set_major_locator(ticker.MaxNLocator(3))

# Increase tick label size
ax.tick_params(axis='x', labelsize=24)
ax.tick_params(axis='y', labelsize=24)
ax.tick_params(axis='z', labelsize=24)

ax.set_title('Threshold Parameter Space - All Datasets Combined', fontsize=32, weight='bold')
ax.legend(fontsize=24, loc='upper right')

# Make pane faces more transparent to improve visibility
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Make grid lines heavier and more visible
ax.grid(True, linestyle='-', linewidth=2.0, alpha=0.6)

# Adjust view angle to make z-label more visible
ax.view_init(elev=20, azim=-35)

plt.tight_layout()

#%%
# 5. BEST PARAMETER COMBINATIONS
# Find the best performing threshold combinations for each dataset

best_params = []
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    
    # Get top 3 parameter combinations
    top_combos = dataset_data.sort_values('csb_value', ascending=False).head(3)
    for _, row in top_combos.iterrows():
        best_params.append({
            'dataset': dataset,
            'cells_max': row['cells_max'],
            'cell_fill': row['cell_fill'],
            'seg_value': row['seg_value'],
            'det_value': row['det_value'],
            'csb_value': row['csb_value'],
        })

best_params_df = pd.DataFrame(best_params)
display(best_params_df)

#%%
# 6. DATASET COMPARISON
# Compare performance across thresholds for different datasets

# Find best threshold for each dataset
best_by_dataset = []
for dataset in df['dataset'].unique():
    dataset_data = df[df['dataset'] == dataset]
    best_row = dataset_data.loc[dataset_data['csb_value'].idxmax()]
    best_by_dataset.append({
        'dataset': dataset,
        'best_cells_max': best_row['cells_max'],
        'best_cell_fill': best_row['cell_fill'],
        'best_csb': best_row['csb_value']
    })

best_df = pd.DataFrame(best_by_dataset)

# Plot best threshold values by dataset
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

ax1.bar(best_df['dataset'], best_df['best_cells_max'], color='skyblue')
ax1.set_ylabel('Best cells_max')
ax1.set_title('Best cells_max Threshold by Dataset')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

ax2.bar(best_df['dataset'], best_df['best_cell_fill'], color='salmon')
ax2.set_ylabel('Best cell_fill')
ax2.set_title('Best cell_fill Threshold by Dataset')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

#%%
# 7. ROBUSTNESS ANALYSIS
# How sensitive is performance to threshold parameters?

# For each dataset, find the range of OP_CSB scores within some percentage of the maximum
for dataset in df['dataset'].unique():
    subset = df[df['dataset'] == dataset]
    max_csb = subset['csb_value'].max()
    threshold_pct = 0.95  # Parameters with OP_CSB within 95% of max
    
    # Filter to high-performing parameter combinations
    high_perf = subset[subset['csb_value'] >= max_csb * threshold_pct]
    
    # Calculate the range of thresholds that yield good performance
    cells_max_range = high_perf['cells_max'].max() - high_perf['cells_max'].min()
    cell_fill_range = high_perf['cell_fill'].max() - high_perf['cell_fill'].min()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(subset['cells_max'], subset['cell_fill'], c=subset['csb_value'], 
               cmap='viridis', s=50, alpha=0.5)
    
    # Highlight high-performing combinations
    plt.scatter(high_perf['cells_max'], high_perf['cell_fill'], color='red', 
               s=100, alpha=0.8, marker='o', edgecolors='black')
    
    plt.title(f'Robust Parameter Region - {dataset}')
    plt.xlabel('cells_max threshold')
    plt.ylabel('cell_fill threshold')
    plt.colorbar(label=r'$\mathrm{OP}_{\mathrm{CSB}}$ Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add annotation with the range information
    plt.annotate(
        f"Robust cells_max range: {cells_max_range:.6f}\n"
        f"Robust cell_fill range: {cell_fill_range:.6f}\n"
        f"Performance within {(1-threshold_pct)*100:.0f}% of max",
        xy=(0.05, 0.05), xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8)
    )
    
    plt.tight_layout()

#%%
# 8. SUMMARY TEXT REPORT
print("=" * 80)
print("THRESHOLD ABLATION STUDY SUMMARY")
print("=" * 80)

# Overall best parameters
best_overall = df.loc[df['csb_value'].idxmax()]
print(f"BEST OVERALL THRESHOLD CONFIGURATION:")
print(f"  Dataset: {best_overall['dataset']}")
print(f"  cells_max: {best_overall['cells_max']:.6f}")
print(f"  cell_fill: {best_overall['cell_fill']:.6f}")
print(f"  OP_CSB Score: {best_overall['csb_value']:.6f}")
print(f"  SEG Score: {best_overall['seg_value']:.6f}")
print(f"  DET Score: {best_overall['det_value']:.6f}")
print()

# Best parameters for each dataset
print("BEST THRESHOLDS BY DATASET:")
for dataset in df['dataset'].unique():
    dataset_best = df[df['dataset'] == dataset].loc[df[df['dataset'] == dataset]['csb_value'].idxmax()]
    print(f"  {dataset}:")
    print(f"    cells_max: {dataset_best['cells_max']:.6f}")
    print(f"    cell_fill: {dataset_best['cell_fill']:.6f}")
    print(f"    OP_CSB Score: {dataset_best['csb_value']:.6f}")
    print(f"    SEG Score: {dataset_best['seg_value']:.6f}")
    print(f"    DET Score: {dataset_best['det_value']:.6f}")
print()

# Find overall best threshold combination across all datasets
# Group by thresholds and get average OP_CSB across all datasets
threshold_combos = {}
for _, row in df.iterrows():
    key = (row['cells_max'], row['cell_fill'])
    if key not in threshold_combos:
        threshold_combos[key] = []
    threshold_combos[key].append(row['csb_value'])

# Calculate average OP_CSB for each threshold combination
avg_csb_by_combo = {}
for combo, csb_values in threshold_combos.items():
    avg_csb_by_combo[combo] = sum(csb_values) / len(csb_values)

# Find overall best thresholds
if avg_csb_by_combo:
    overall_best_combo = max(avg_csb_by_combo.items(), key=lambda x: x[1])
    overall_best_cells_max, overall_best_cell_fill = overall_best_combo[0]
    overall_best_csb = overall_best_combo[1]
    
    # Calculate average SEG and DET scores for this threshold combination
    best_combo_rows = df[(df['cells_max'] == overall_best_cells_max) & 
                         (df['cell_fill'] == overall_best_cell_fill)]
    
    overall_best_seg = best_combo_rows['seg_value'].mean()
    overall_best_det = best_combo_rows['det_value'].mean()
    
    print("OVERALL BEST THRESHOLDS ACROSS ALL DATASETS:")
    print(f"  cells_max: {overall_best_cells_max:.6f}")
    print(f"  cell_fill: {overall_best_cell_fill:.6f}")
    print(f"  Average OP_CSB: {overall_best_csb:.6f}")
    print(f"  Average SEG: {overall_best_seg:.6f}")
    print(f"  Average DET: {overall_best_det:.6f}")

print("=" * 80) 