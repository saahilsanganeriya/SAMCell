#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
import matplotlib.ticker as ticker

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
plt.rcParams['mathtext.default'] = 'regular'  # Use regular font for math

#%%
# Load pretraining and per epoch data
print("Loading pretraining and per epoch data...")
pretraining_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell-paper/src/ablations_final/study_4_pretraining.csv"
per_epoch_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell-paper/src/ablations_final/study_5_per_epoch.csv"

# Load pretraining data
if os.path.exists(pretraining_file):
    df_pretraining = pd.read_csv(pretraining_file)
    print(f"Loaded pretraining data from {pretraining_file} with {len(df_pretraining)} entries")
else:
    print(f"File {pretraining_file} not found.")
    df_pretraining = pd.DataFrame()

# Try to load the per epoch data
if os.path.exists(per_epoch_file):
    # Read the CSV file line by line to handle inconsistent column count
    with open(per_epoch_file, 'r') as f:
        lines = f.readlines()
    
    # Get headers from the first line
    headers = lines[0].strip().split(',')
    
    # Create a list to store data
    data_rows = []
    
    # Process each line starting from the second line (index 1)
    for line in lines[1:]:
        # Split the line by commas
        values = line.strip().split(',')
        
        # Check if this row has an extra column (epoch value at the end)
        if len(values) > len(headers):
            # The last value is the epoch
            epoch_value = values[-1]
            # Remove the last value to match header count
            values = values[:-1]
            # Create a row dict with headers
            row_dict = dict(zip(headers, values))
            # Add epoch separately
            row_dict['epoch'] = epoch_value
        else:
            # Create a row dict with headers
            row_dict = dict(zip(headers, values))
            # No explicit epoch, will extract from checkpoint name later
        
        data_rows.append(row_dict)
    
    # Convert to DataFrame
    df_epochs = pd.DataFrame(data_rows)
    print(f"Loaded per epoch data from {per_epoch_file} with {len(df_epochs)} entries")
else:
    print(f"File {per_epoch_file} not found.")
    df_epochs = pd.DataFrame()

#%%
# Extract epoch number from checkpoint name
if not df_epochs.empty and 'checkpoint_name' in df_epochs.columns:
    def extract_epoch(checkpoint_name):
        match = re.search(r'epoch-(\d+)', str(checkpoint_name))
        if match:
            return int(match.group(1))
        return None

    # Extract epochs from checkpoint names if not already present
    if 'epoch' not in df_epochs.columns:
        df_epochs['epoch'] = df_epochs['checkpoint_name'].apply(extract_epoch)
    else:
        # If epoch column exists but might be a string, convert to int
        df_epochs['epoch'] = pd.to_numeric(df_epochs['epoch'], errors='coerce')
    
    # Remove rows with missing epoch values
    df_epochs = df_epochs.dropna(subset=['epoch'])
    
    # Make sure epoch is an integer
    df_epochs['epoch'] = df_epochs['epoch'].astype(int)

    # Remove rows with epoch value 35
    df_epochs = df_epochs[df_epochs['epoch'] != 35]
    print(f"Extracted epoch numbers. Available epochs: {sorted(df_epochs['epoch'].unique())}")

#%%
# Label whether model is pretrained or random weights in both dataframes
if not df_epochs.empty:
    # Determine whether model is pretrained based on run_id or run_name
    if 'run_name' in df_epochs.columns:
        df_epochs['is_pretrained'] = ~df_epochs['run_name'].str.contains('random-weights')
    elif 'run_id' in df_epochs.columns:
        # Assuming lpaw1rdp is random weights and xeaxlskz is pretrained
        df_epochs['is_pretrained'] = (df_epochs['run_id'] != 'lpaw1rdp')
    
    df_epochs['model_label'] = df_epochs['is_pretrained'].map({True: 'Pretrained', False: 'Random Weights'})

if not df_pretraining.empty:
    df_pretraining['is_pretrained'] = ~df_pretraining['run_name'].str.contains('random-weights')
    df_pretraining['model_label'] = df_pretraining['is_pretrained'].map({True: 'Pretrained', False: 'Random Weights'})

#%%
# Filter out CellPose-test dataset from both dataframes
if not df_pretraining.empty:
    df_pretraining_filtered = df_pretraining[df_pretraining['dataset'] != 'CellPose-test']
    print(f"Filtered pretraining dataset shape: {df_pretraining_filtered.shape}")
    print(f"Remaining datasets in pretraining: {df_pretraining_filtered['dataset'].unique()}")
else:
    df_pretraining_filtered = pd.DataFrame()

if not df_epochs.empty:
    df_epochs_filtered = df_epochs[df_epochs['dataset'] != 'CellPose-test']
    print(f"Filtered epochs dataset shape: {df_epochs_filtered.shape}")
    print(f"Remaining datasets in epochs: {df_epochs_filtered['dataset'].unique()}")
    
    if 'epoch' in df_epochs_filtered.columns:
        print(f"Available epochs: {sorted(df_epochs_filtered['epoch'].unique())}")
else:
    df_epochs_filtered = pd.DataFrame()

# Convert all metric columns to float
for col in ['seg_value', 'det_value', 'csb_value']:
    if col in df_epochs_filtered.columns:
        df_epochs_filtered[col] = pd.to_numeric(df_epochs_filtered[col], errors='coerce')
    if col in df_pretraining_filtered.columns:
        df_pretraining_filtered[col] = pd.to_numeric(df_pretraining_filtered[col], errors='coerce')

# Display dataframe sample for debugging
if not df_epochs_filtered.empty:
    print("\nSample of epochs data:")
    print(df_epochs_filtered.head())

#%%
# Define metric display names
metric_display_names = {
    'seg_value': 'SEG',
    'det_value': 'DET',
    'csb_value': r'$\mathrm{OP}_{\mathrm{CSB}}$'
}

#%%
# 1. Plot OP_CSB over epochs for pretrained vs random initialization
def plot_op_csb_over_epochs(df_epochs, save_path=None):
    """Plot OP_CSB over epochs for pretrained and random initialization, with separate plots for each dataset."""
    
    if df_epochs.empty:
        print("No epoch data available for plotting.")
        return None
    
    # Get unique datasets and model types
    datasets = sorted(df_epochs['dataset'].unique())
    model_types = sorted(df_epochs['model_label'].unique())
    
    # Preprocessing: Add 1 to all epoch numbers (correcting offset)
    df_epochs = df_epochs.copy()
    df_epochs['epoch'] = df_epochs['epoch'] + 1
    
    # Add epoch 0 datapoints with zeros for all metrics
    epoch0_rows = []
    for dataset in datasets:
        for model_label in model_types:
            # Create a new row for epoch 0 with zeros for metrics
            epoch0_row = {
                'dataset': dataset,
                'model_label': model_label,
                'epoch': 0,
                'csb_value': 0.0,
                'seg_value': 0.0,
                'det_value': 0.0,
                'is_pretrained': model_label == 'Pretrained'
            }
            epoch0_rows.append(epoch0_row)
    
    # Add the epoch 0 rows to the dataframe
    epoch0_df = pd.DataFrame(epoch0_rows)
    df_epochs = pd.concat([df_epochs, epoch0_df], ignore_index=True)
    
    # Set up colors and markers
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D']
    
    # Font settings for research paper
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['legend.fontsize'] = 18
    
    # Create a separate plot for each dataset
    plots = []
    for i, dataset in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create a list to store handles for the legend
        legend_handles = []
        
        # Filter data for this dataset
        dataset_data = df_epochs[df_epochs['dataset'] == dataset]
        
        # Find min and max y values for setting axis limits
        min_csb = 0  # Start from 0 since we added zeros
        max_csb = dataset_data['csb_value'].max() * 1.05  # Add 5% padding above
        
        # Create directory for .dat files
        dat_dir = "ablations_vis/data"
        os.makedirs(dat_dir, exist_ok=True)
        
        # Plot lines for each model type
        for j, model_type in enumerate(model_types):
            # Filter data for this model type
            data = dataset_data[dataset_data['model_label'] == model_type]
            
            if not data.empty:
                # Sort by epoch to ensure correct line
                data = data.sort_values('epoch')
                
                # Create file name: dataset_modeltype.dat
                dataset_name = dataset.replace(' ', '_').lower()
                model_name = model_type.replace(' ', '_').lower()
                dat_file = os.path.join(dat_dir, f"{dataset_name}_{model_name}.dat")
                
                # Create .dat file with header and data
                with open(dat_file, 'w') as f:
                    f.write("# epoch op_csb_score\n")
                    # Add epoch 0 with value 0
                    f.write("0 0\n")
                    # Write all data rows
                    for _, row in data.iterrows():
                        f.write(f"{int(row['epoch'])} {row['csb_value']}\n")
                
                print(f"Created data file: {dat_file}")
                
                # Plot the line
                line, = ax.plot(data['epoch'], data['csb_value'], 
                               marker=markers[j % len(markers)],
                               linestyle='-' if model_type == 'Pretrained' else '--',
                               linewidth=3,  # Thicker lines
                               markersize=14,  # Larger markers
                               color=colors[j % len(colors)],
                               label=f"{model_type}")
                
                # Add legend handle
                legend_handles.append(line)
        
        # Customize the plot
        ax.set_xlabel('Epoch', fontsize=22, fontweight='bold')
        ax.set_ylabel(r'$\mathrm{OP}_{\mathrm{CSB}}$', fontsize=22, fontweight='bold')
        ax.set_title(f'{dataset}', fontsize=24, fontweight='bold')
        
        # Set x-axis ticks to be integers with fewer ticks
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        
        # Set y-axis with data-driven range and fewer ticks
        ax.set_ylim(min_csb, max_csb)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add legend
        ax.legend(handles=legend_handles, loc='lower right', fontsize=18)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Create a dataset-specific filename
            dataset_filename = save_path.replace('.png', f'_{dataset.replace(" ", "_")}.png')
            plt.savefig(dataset_filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {dataset_filename}")
        
        plots.append(fig)
    
    # Return the list of figures
    return plots

#%%
# 2. Create a metric comparison bar chart for the final epoch
def plot_final_metrics_comparison(df_epochs, df_pretraining, metric='csb_value', save_path=None):
    """Plot final metrics comparison for pretrained and random initialization."""
    
    if df_epochs.empty and df_pretraining.empty:
        print("No data available for plotting.")
        return None
    
    # If we have epoch data, use the highest epoch for each model as "final"
    if not df_epochs.empty:
        # Get the highest epoch for each dataset and model type
        final_epochs = df_epochs.groupby(['dataset', 'model_label'])['epoch'].max().reset_index()
        
        # Merge with the main dataframe to get only the final epoch rows
        df_final = pd.merge(df_epochs, final_epochs, on=['dataset', 'model_label', 'epoch'])
    else:
        # If no epoch data, use pretraining data
        df_final = df_pretraining.copy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get unique datasets and model types
    datasets = sorted(df_final['dataset'].unique())
    model_types = sorted(df_final['model_label'].unique())
    
    # Set up the x positions
    x = np.arange(len(datasets))
    width = 0.35  # Width of bars
    
    # Plot bars for each model type
    for i, model_type in enumerate(model_types):
        # Get values for this model type
        values = []
        for dataset in datasets:
            data = df_final[(df_final['dataset'] == dataset) & 
                            (df_final['model_label'] == model_type)]
            if not data.empty:
                values.append(data[metric].iloc[0])
            else:
                values.append(0)
        
        # Plot bar
        bars = ax.bar(x + (i - 0.5 + 0.5/len(model_types)) * width, values, width/(len(model_types)), 
                     label=model_type,
                     alpha=0.8)
        
        # Add value annotations
        for j, v in enumerate(values):
            if v > 0:
                ax.text(x[j] + (i - 0.5 + 0.5/len(model_types)) * width, v + 0.02, 
                       f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel(f"{metric_display_names.get(metric, metric)} Score", fontsize=14)
    ax.set_title(f'Final {metric_display_names.get(metric, metric)} Score Comparison\nPretrained vs Random Initialization', 
                fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#%%
# 3. Create a radar chart comparing all metrics
def plot_radar_chart(df_epochs, df_pretraining, save_path=None):
    """Create a radar chart comparing SEG, DET, and OP_CSB for different models."""
    
    if df_epochs.empty and df_pretraining.empty:
        print("No data available for plotting.")
        return None
    
    # If we have epoch data, use the highest epoch for each model as "final"
    if not df_epochs.empty:
        # Get the highest epoch for each dataset and model type
        final_epochs = df_epochs.groupby(['dataset', 'model_label'])['epoch'].max().reset_index()
        
        # Merge with the main dataframe to get only the final epoch rows
        df_final = pd.merge(df_epochs, final_epochs, on=['dataset', 'model_label', 'epoch'])
    else:
        # If no epoch data, use pretraining data
        df_final = df_pretraining.copy()
    
    # Set up the metrics for the radar chart
    metrics = ['seg_value', 'det_value', 'csb_value']
    num_metrics = len(metrics)
    
    # Get unique datasets and model types
    datasets = sorted(df_final['dataset'].unique())
    model_types = sorted(df_final['model_label'].unique())
    
    # Create a figure for the radar chart with one subplot per dataset
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 8), 
                           subplot_kw=dict(polar=True))
    
    # If only one dataset, make axes an array
    if len(datasets) == 1:
        axes = [axes]
    
    # Set the angles for each metric (evenly spaced around the circle)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    # Make the plot a complete circle
    angles += angles[:1]
    
    # Set up colors
    colors = plt.cm.tab10.colors
    
    # Plot data for each dataset
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        # Set up the angles for the radar chart
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Set the labels for each metric
        metric_labels = [metric_display_names.get(m, m) for m in metrics]
        # Add the first label again to complete the circle
        metric_labels_plot = metric_labels + [metric_labels[0]]
        plt.xticks(angles, metric_labels_plot, fontsize=12)
        
        # Set y limit
        ax.set_ylim(0, 1)
        
        # Plot data for each model type
        for j, model_type in enumerate(model_types):
            # Get data for this dataset and model type
            data = df_final[(df_final['dataset'] == dataset) & 
                           (df_final['model_label'] == model_type)]
            
            if not data.empty:
                # Get the values for each metric
                values = [data[metric].iloc[0] for metric in metrics]
                # Add the first value again to complete the circle
                values += values[:1]
                
                # Plot the line
                ax.plot(angles, values, 'o-', linewidth=2, label=model_type, 
                       color=colors[j % len(colors)], markersize=8)
                
                # Fill the area
                ax.fill(angles, values, alpha=0.1, color=colors[j % len(colors)])
        
        # Add title for this subplot
        ax.set_title(dataset, fontsize=14)
        
        # Add legend to the first subplot only
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.suptitle('Radar Chart: Metric Comparison', fontsize=16, y=1.05)
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#%%
# 4. Create a line plot for all three metrics over epochs
def plot_all_metrics_over_epochs(df_epochs, dataset_filter=None, save_path=None):
    """Plot all three metrics (SEG, DET, OP_CSB) over epochs for a specific dataset."""
    
    if df_epochs.empty:
        print("No epoch data available for plotting.")
        return None
    
    # Filter by dataset if specified
    if dataset_filter:
        df_plot = df_epochs[df_epochs['dataset'] == dataset_filter]
        if df_plot.empty:
            print(f"No data available for dataset {dataset_filter}")
            return None
        title_suffix = f" - {dataset_filter}"
    else:
        # Use all datasets, averaging across them
        df_plot = df_epochs
        title_suffix = " - All Datasets"
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get model types
    model_types = sorted(df_plot['model_label'].unique())
    
    # Define metrics to plot
    metrics = ['seg_value', 'det_value', 'csb_value']
    
    # Set up colors and line styles
    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    # Plot lines for each model type and metric
    for i, model_type in enumerate(model_types):
        model_data = df_plot[df_plot['model_label'] == model_type]
        
        if not model_data.empty:
            # Only select numeric columns for groupby mean
            numeric_model_data = model_data[['epoch'] + metrics].copy()
            
            try:
                # Make sure 'epoch' is numeric
                numeric_model_data['epoch'] = pd.to_numeric(numeric_model_data['epoch'], errors='coerce')
                
                # Group by epoch and calculate mean for each metric
                model_by_epoch = numeric_model_data.groupby('epoch').mean().reset_index()
                
                # Sort by epoch
                model_by_epoch = model_by_epoch.sort_values('epoch')
                
                # Plot each metric
                for j, metric in enumerate(metrics):
                    ax.plot(model_by_epoch['epoch'], model_by_epoch[metric], 
                           marker=markers[j % len(markers)],
                           linestyle=line_styles[j % len(line_styles)],
                           color=colors[i % len(colors)],
                           linewidth=2,
                           label=f"{model_type} - {metric_display_names.get(metric, metric)}")
            except Exception as e:
                print(f"Error processing data for {model_type}: {e}")
                # If groupby fails, just plot individual points
                for j, metric in enumerate(metrics):
                    ax.scatter(numeric_model_data['epoch'], numeric_model_data[metric], 
                              marker=markers[j % len(markers)],
                              color=colors[i % len(colors)],
                              label=f"{model_type} - {metric_display_names.get(metric, metric)} (raw)")
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title(f'Metrics Over Training Epochs{title_suffix}', fontsize=16)
    
    # Set x-axis ticks to be integers if possible
    try:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    except:
        pass
    
    # Set y-axis limits
    ax.set_ylim(0, 1.0)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#%%
# 5. Create a heatmap of improvement over epochs
def plot_improvement_heatmap(df_epochs, save_path=None):
    """Create a heatmap showing improvement in OP_CSB over epochs compared to initial value."""
    
    if df_epochs.empty:
        print("No epoch data available for plotting.")
        return None
    
    # Ensure epoch is numeric
    df_epochs_numeric = df_epochs.copy()
    df_epochs_numeric['epoch'] = pd.to_numeric(df_epochs_numeric['epoch'], errors='coerce')
    
    # Ensure we have valid epoch values
    df_epochs_numeric = df_epochs_numeric.dropna(subset=['epoch'])
    
    if df_epochs_numeric.empty:
        print("No valid epoch data after conversion to numeric.")
        return None
    
    # Create a figure for the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique datasets and model types
    datasets = sorted(df_epochs_numeric['dataset'].unique())
    model_types = sorted(df_epochs_numeric['model_label'].unique())
    
    # Create a DataFrame to store improvement values
    improvement_data = []
    
    # Calculate improvement for each dataset and model type
    for dataset in datasets:
        for model_type in model_types:
            # Filter data for this dataset and model type
            data = df_epochs_numeric[(df_epochs_numeric['dataset'] == dataset) & 
                               (df_epochs_numeric['model_label'] == model_type)]
            
            if not data.empty:
                # Sort by epoch
                data = data.sort_values('epoch')
                
                # Get the initial value (either epoch 0 or the first available)
                initial_epoch = data['epoch'].min()
                initial_value = data[data['epoch'] == initial_epoch]['csb_value'].iloc[0]
                
                # Calculate improvement for each epoch
                for _, row in data.iterrows():
                    epoch = row['epoch']
                    value = row['csb_value']
                    
                    # Calculate absolute and percentage improvement
                    abs_improvement = value - initial_value
                    pct_improvement = (abs_improvement / initial_value) * 100 if initial_value > 0 else 0
                    
                    improvement_data.append({
                        'dataset': dataset,
                        'model_label': model_type,
                        'epoch': int(epoch),  # Convert to int for better display
                        'absolute_improvement': abs_improvement,
                        'percentage_improvement': pct_improvement
                    })
    
    # Convert to DataFrame
    improvement_df = pd.DataFrame(improvement_data)
    
    if improvement_df.empty:
        print("No improvement data could be calculated.")
        return None
    
    # Create index labels that combine dataset and model_label
    improvement_df['dataset_model'] = improvement_df['dataset'] + ' - ' + improvement_df['model_label']
    
    # Create pivot table for the heatmap
    try:
        pivot = improvement_df.pivot_table(
            index='dataset_model',
            columns='epoch',
            values='percentage_improvement',
            aggfunc='mean'  # In case there are duplicates
        )
        
        # Create the heatmap
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.1f', ax=ax,
                   cbar_kws={'label': 'OP_CSB Improvement (%)'})
        
        # Customize the plot
        ax.set_title('OP_CSB Improvement Over Epochs (% relative to initial value)', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Dataset - Model Type', fontsize=14)
        
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None

#%%
# Create output directory
save_dir = "paper/pretraining_epochs_ablation"
os.makedirs(save_dir, exist_ok=True)

# Generate all visualizations
# 1. OP_CSB over epochs
op_csb_path = os.path.join(save_dir, "op_csb_over_epochs.png")
plot_op_csb_over_epochs(df_epochs_filtered, op_csb_path)

# 2. Final metrics comparison
final_metrics_path = os.path.join(save_dir, "final_metrics_comparison.png")
plot_final_metrics_comparison(df_epochs_filtered, df_pretraining_filtered, 'csb_value', final_metrics_path)

# Additional metric comparisons
plot_final_metrics_comparison(df_epochs_filtered, df_pretraining_filtered, 'seg_value', 
                             os.path.join(save_dir, "final_seg_comparison.png"))
plot_final_metrics_comparison(df_epochs_filtered, df_pretraining_filtered, 'det_value', 
                             os.path.join(save_dir, "final_det_comparison.png"))

# 3. Radar chart
radar_path = os.path.join(save_dir, "radar_chart_comparison.png")
plot_radar_chart(df_epochs_filtered, df_pretraining_filtered, radar_path)

# 4. All metrics over epochs
for dataset in df_epochs_filtered['dataset'].unique():
    metrics_path = os.path.join(save_dir, f"all_metrics_over_epochs_{dataset}.png")
    plot_all_metrics_over_epochs(df_epochs_filtered, dataset, metrics_path)

# 5. Improvement heatmap
heatmap_path = os.path.join(save_dir, "improvement_heatmap.png")
plot_improvement_heatmap(df_epochs_filtered, heatmap_path)

#%%
# Summary analysis
print("\n" + "="*80)
print("PRETRAINING AND TRAINING EPOCHS STUDY SUMMARY")
print("="*80)

# Reset index for the dataframes to avoid duplicates
df_epochs_filtered = df_epochs_filtered.reset_index(drop=True)
df_pretraining_filtered = df_pretraining_filtered.reset_index(drop=True)

# Summary for pretraining effect
if not df_pretraining_filtered.empty:
    print("\nPretraining Effect on Final Performance:")
    print("-" * 40)
    
    # Group by model_label and calculate average metrics
    pretrain_summary = df_pretraining_filtered.groupby('model_label').agg({
        'seg_value': 'mean',
        'det_value': 'mean',
        'csb_value': 'mean'
    }).reset_index()
    
    # Display summary for each model type
    for _, row in pretrain_summary.iterrows():
        model_type = row['model_label']
        print(f"\nModel: {model_type}")
        for metric in ['seg_value', 'det_value', 'csb_value']:
            print(f"  Average {metric_display_names.get(metric, metric)}: {row[metric]:.6f}")
    
    # Calculate improvement from pretraining
    if len(pretrain_summary) > 1:
        pretrained_row = pretrain_summary[pretrain_summary['model_label'] == 'Pretrained']
        random_row = pretrain_summary[pretrain_summary['model_label'] == 'Random Weights']
        
        if not pretrained_row.empty and not random_row.empty:
            print("\nImprovement from Pretraining:")
            for metric in ['seg_value', 'det_value', 'csb_value']:
                pretrained_val = pretrained_row[metric].iloc[0]
                random_val = random_row[metric].iloc[0]
                improvement = pretrained_val - random_val
                pct_improvement = (improvement / random_val) * 100 if random_val > 0 else 0
                
                print(f"  {metric_display_names.get(metric, metric)}: {improvement:.6f} ({pct_improvement:.2f}%)")

# Summary for training epochs
if not df_epochs_filtered.empty:
    print("\nTraining Epochs Effect:")
    print("-" * 40)
    
    # Group by model_label and epoch, and calculate average metrics
    epoch_summary = df_epochs_filtered.groupby(['model_label', 'epoch']).agg({
        'seg_value': 'mean',
        'det_value': 'mean',
        'csb_value': 'mean'
    }).reset_index()
    
    # For each model type, show improvement from first to last epoch
    for model_type in epoch_summary['model_label'].unique():
        model_data = epoch_summary[epoch_summary['model_label'] == model_type].sort_values('epoch')
        
        if len(model_data) > 1:
            first_epoch = model_data['epoch'].min()
            last_epoch = model_data['epoch'].max()
            
            first_row = model_data[model_data['epoch'] == first_epoch]
            last_row = model_data[model_data['epoch'] == last_epoch]
            
            print(f"\nModel: {model_type}")
            print(f"  Improvement from epoch {first_epoch} to {last_epoch}:")
            
            for metric in ['seg_value', 'det_value', 'csb_value']:
                first_val = first_row[metric].iloc[0]
                last_val = last_row[metric].iloc[0]
                improvement = last_val - first_val
                pct_improvement = (improvement / first_val) * 100 if first_val > 0 else 0
                
                print(f"  {metric_display_names.get(metric, metric)}: {improvement:.6f} ({pct_improvement:.2f}%)")

# Overall best configuration
if not df_epochs_filtered.empty:
    # Find the row with the highest csb_value
    max_csb_idx = df_epochs_filtered['csb_value'].idxmax()
    # Get the actual row
    best_epoch_row = df_epochs_filtered.iloc[max_csb_idx]
    
    print("\nBest Overall Configuration:")
    print(f"  Model: {best_epoch_row['model_label']}")
    print(f"  Dataset: {best_epoch_row['dataset']}")
    print(f"  Epoch: {best_epoch_row['epoch']}")
    print(f"  OP_CSB Score: {best_epoch_row['csb_value']:.6f}")
    print(f"  SEG Score: {best_epoch_row['seg_value']:.6f}")
    print(f"  DET Score: {best_epoch_row['det_value']:.6f}")
elif not df_pretraining_filtered.empty:
    # Find the row with the highest csb_value
    max_csb_idx = df_pretraining_filtered['csb_value'].idxmax()
    # Get the actual row
    best_pretrain_row = df_pretraining_filtered.iloc[max_csb_idx]
    
    print("\nBest Overall Configuration:")
    print(f"  Model: {best_pretrain_row['model_label']}")
    print(f"  Dataset: {best_pretrain_row['dataset']}")
    print(f"  OP_CSB Score: {best_pretrain_row['csb_value']:.6f}")
    print(f"  SEG Score: {best_pretrain_row['seg_value']:.6f}")
    print(f"  DET Score: {best_pretrain_row['det_value']:.6f}")

print("="*80) 