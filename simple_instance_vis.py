#!/usr/bin/env python3
"""
Simple Instance Visualization Script for YOLO Dataset
Creates a single bar graph showing instances per class across train/val/test splits.
"""

import os
import yaml
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def load_config(data_yaml_path='data.yaml'):
    """Load configuration from data.yaml file."""
    with open(data_yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['names'], config['nc']

def count_instances_in_split(split_name, nc):
    """Count instances for each class in a specific split."""
    labels_dir = os.path.join(split_name, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory {labels_dir} not found!")
        return Counter()
    
    class_counts = Counter()
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            # Handle both integer and float class IDs
                            class_id = int(float(parts[0]))
                            if 0 <= class_id < nc:
                                class_counts[class_id] += 1
                        except ValueError:
                            continue
        
        except Exception:
            continue
    
    return class_counts

def main():
    """Main function to create the bar graph."""
    # Configuration variables - modify these as needed
    TEXT_SIZE = 24  # Adjust this value to change all text sizes
    BAR_WIDTH = 0.35  # Width of bars (smaller = thinner bars)
    GROUP_SPACING = 0.8  # Spacing between groups (smaller = closer together)
    FIGURE_SIZE = (12, 9)  # Figure dimensions (width, height)
    BAR_CORNER_RADIUS = 0.3  # Radius for rounded corners (0 = square, higher = more rounded)
    BACKGROUND_COLOR = '#F5F5F5'  # Light gray background color
    
    # Text labels (modify these as needed)
    PLOT_TITLE = "Instance Distribution Across Dataset Splits"
    X_AXIS_LABEL = "Dataset Split"
    Y_AXIS_LABEL = "Number of Instances"
    
    # Load configuration
    class_names, nc = load_config()
    splits = ['train', 'val', 'test']
    
    print(f"Classes: {class_names}")
    print("Counting instances...")
    
    # Collect data for all splits
    data = {}
    for split in splits:
        print(f"Processing {split}...")
        counts = count_instances_in_split(split, nc)
        data[split] = [counts.get(i, 0) for i in range(nc)]
    
    # Print statistics
    print("\nStatistics:")
    for split in splits:
        total = sum(data[split])
        print(f"{split}: {data[split]} (total: {total})")
    
    # Create the bar graph
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor(BACKGROUND_COLOR)  # Set figure background color
    ax.set_facecolor(BACKGROUND_COLOR)  # Set axes background color
    
    x = np.arange(len(splits)) * GROUP_SPACING  # Split positions with custom spacing
    
    # Get colors for the bars
    colors = plt.cm.Set2.colors[:2]  # Use Set2 colormap for nice colors
    
    # Create bars for each class with rounded corners
    bar_data1 = [data[split][0] for split in splits]
    bar_data2 = [data[split][1] for split in splits]
    
    # Create bars with proper colors for legend
    bars1 = ax.bar(x - BAR_WIDTH/2, bar_data1, BAR_WIDTH, 
                   label=class_names[0], color=colors[0], alpha=0.8)
    bars2 = ax.bar(x + BAR_WIDTH/2, bar_data2, BAR_WIDTH, 
                   label=class_names[1], color=colors[1], alpha=0.8)
    
    # Apply rounded corners to bars while keeping original colors
    for bar in bars1:
        bar.set_edgecolor('white')
        bar.set_linewidth(1)
        # Create rounded rectangle patch
        x_pos, y_pos = bar.get_x(), bar.get_y()
        width, height = bar.get_width(), bar.get_height()
        rounded_rect = patches.FancyBboxPatch((x_pos, y_pos), width, height,
                                            boxstyle=f"round,pad=0,rounding_size={BAR_CORNER_RADIUS}",
                                            facecolor=colors[0],
                                            edgecolor='white', linewidth=1, alpha=0.8)
        ax.add_patch(rounded_rect)
        bar.set_alpha(0)  # Hide original bar, but keep for legend
    
    for bar in bars2:
        bar.set_edgecolor('white')
        bar.set_linewidth(1)
        # Create rounded rectangle patch
        x_pos, y_pos = bar.get_x(), bar.get_y()
        width, height = bar.get_width(), bar.get_height()
        rounded_rect = patches.FancyBboxPatch((x_pos, y_pos), width, height,
                                            boxstyle=f"round,pad=0,rounding_size={BAR_CORNER_RADIUS}",
                                            facecolor=colors[1],
                                            edgecolor='white', linewidth=1, alpha=0.8)
        ax.add_patch(rounded_rect)
        bar.set_alpha(0)  # Hide original bar, but keep for legend
    
    # Create custom legend with correct colors
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=0.8, label=class_names[0]),
                      plt.Rectangle((0,0),1,1, facecolor=colors[1], alpha=0.8, label=class_names[1])]
    ax.legend(handles=legend_elements, fontsize=TEXT_SIZE)
    
    # Add title and axis labels
    ax.set_title(PLOT_TITLE, fontsize=TEXT_SIZE + 2, fontweight='bold', pad=20)
    ax.set_xlabel(X_AXIS_LABEL, fontsize=TEXT_SIZE, labelpad=10)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=TEXT_SIZE, labelpad=10)
    
    # Customize the graph - bigger text for embedding
    ax.set_xticks(x)
    ax.set_xticklabels([split.capitalize() for split in splits], fontsize=TEXT_SIZE)
    ax.tick_params(axis='y', labelsize=TEXT_SIZE)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot with background color (not transparent)
    output_path = 'instance_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    print(f"\nGraph saved as: {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()