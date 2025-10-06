import pandas as pd
import numpy as np
import os

# Load threshold ablation data
threshold_file = "/Users/saahilsanganeriya/Documents/Saahil/PBL/SAMCell-paper/src/ablations_final/ablation_thresholds_final_old.csv"
df = pd.read_csv(threshold_file)

# Filter out cell_fill = 0 rows and CellPose-test dataset
df = df[df['cell_fill'] > 0]
df = df[df['dataset'] != 'CellPose-test']

# Create output directory if it doesn't exist
os.makedirs('ablations_vis', exist_ok=True)

# Filter for specific datasets
datasets = ['PBL_HEK', 'PBL_N2A']

for dataset in datasets:
    # Filter data for this dataset
    dataset_data = df[df['dataset'] == dataset]
    
    # Find the best point
    best_idx = dataset_data['csb_value'].idxmax()
    best_point = dataset_data.loc[best_idx]
    
    # Create LaTeX data file
    output_file = f'data_{dataset.lower()}.dat'
    
    # Save data in format suitable for pgfplots
    with open(output_file, 'w') as f:
        f.write('cells_max cell_fill csb\n')  # Header
        for _, row in dataset_data.iterrows():
            f.write(f'{row["cells_max"]:.6f} {row["cell_fill"]:.6f} {row["csb_value"]:.6f}\n')
    
    # Save best point info
    best_point_file = f'best_point_{dataset.lower()}.dat'
    with open(best_point_file, 'w') as f:
        f.write('cells_max cell_fill csb\n')
        f.write(f'{best_point["cells_max"]:.6f} {best_point["cell_fill"]:.6f} {best_point["csb_value"]:.6f}\n')
    
    # Print summary for this dataset
    print(f"\nDataset: {dataset}")
    print(f"Data points: {len(dataset_data)}")
    print(f"Best point:")
    print(f"  CSB: {best_point['csb_value']:.6f}")
    print(f"  CELLS_MAX: {best_point['cells_max']:.6f}")
    print(f"  CELL_FILL: {best_point['cell_fill']:.6f}")
    print(f"Data saved to: {output_file}")
    print(f"Best point saved to: {best_point_file}")

# Also save min/max values for proper scaling
summary_data = {
    'csb_min': df['csb_value'].min(),
    'csb_max': df['csb_value'].max(),
    'cells_max_min': df['cells_max'].min(),
    'cells_max_max': df['cells_max'].max(),
    'cell_fill_min': df['cell_fill'].min(),
    'cell_fill_max': df['cell_fill'].max(),
}

with open('plot_ranges.dat', 'w') as f:
    for key, value in summary_data.items():
        f.write(f'{key}={value:.6f}\n') 