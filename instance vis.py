#!/usr/bin/env python3
"""
Instance Visualization Script for YOLO Dataset
Analyzes and displays the distribution of instances for each class across train/val/test splits.
"""

import os
import yaml
import glob
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class DatasetAnalyzer:
    def __init__(self, data_yaml_path='data.yaml'):
        """
        Initialize the dataset analyzer.
        
        Args:
            data_yaml_path (str): Path to the data.yaml file
        """
        self.data_yaml_path = data_yaml_path
        self.class_names = []
        self.nc = 0
        self.splits = ['train', 'val', 'test']
        self.instance_counts = defaultdict(lambda: defaultdict(int))
        self.file_counts = defaultdict(int)
        
        self.load_config()
        self.analyze_dataset()
    
    def load_config(self):
        """Load configuration from data.yaml file."""
        if not os.path.exists(self.data_yaml_path):
            raise FileNotFoundError(f"Configuration file {self.data_yaml_path} not found!")
        
        with open(self.data_yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.class_names = config['names']
        self.nc = config['nc']
        
        print(f"Dataset Configuration:")
        print(f"Number of classes: {self.nc}")
        print(f"Class names: {self.class_names}")
        print("-" * 50)
    
    def count_instances_in_split(self, split_name):
        """
        Count instances for each class in a specific split.
        
        Args:
            split_name (str): Name of the split (train/val/test)
        
        Returns:
            dict: Dictionary with class counts and file count
        """
        labels_dir = os.path.join(split_name, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory {labels_dir} not found!")
            return {'class_counts': Counter(), 'file_count': 0}
        
        class_counts = Counter()
        file_count = 0
        
        # Get all label files
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        
        for label_file in label_files:
            file_count += 1
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) >= 5:  # YOLO format: class_id x_center y_center width height
                            try:
                                # Handle both integer and float class IDs
                                class_id = int(float(parts[0]))
                                if 0 <= class_id < self.nc:
                                    class_counts[class_id] += 1
                                else:
                                    print(f"Warning: Invalid class_id {class_id} in {label_file}")
                            except ValueError:
                                print(f"Warning: Cannot parse class_id '{parts[0]}' in {label_file}")
            
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
        
        return {'class_counts': class_counts, 'file_count': file_count}
    
    def analyze_dataset(self):
        """Analyze the entire dataset and collect statistics."""
        print("Analyzing dataset...")
        
        for split in self.splits:
            print(f"Processing {split} split...")
            result = self.count_instances_in_split(split)
            
            self.file_counts[split] = result['file_count']
            
            for class_id, count in result['class_counts'].items():
                if class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                    self.instance_counts[split][class_name] = count
        
        print("Analysis complete!")
        print("-" * 50)
    
    def print_statistics(self):
        """Print detailed statistics for each split and class."""
        print("DATASET STATISTICS")
        print("=" * 60)
        
        # Create a summary table
        data = []
        total_instances_per_class = defaultdict(int)
        
        for split in self.splits:
            for class_name in self.class_names:
                count = self.instance_counts[split].get(class_name, 0)
                data.append({
                    'Split': split,
                    'Class': class_name,
                    'Instances': count,
                    'Files': self.file_counts[split]
                })
                total_instances_per_class[class_name] += count
        
        # Print split-wise statistics
        for split in self.splits:
            print(f"\n{split.upper()} SPLIT:")
            print(f"  Total files: {self.file_counts[split]}")
            total_instances_in_split = 0
            
            for class_name in self.class_names:
                count = self.instance_counts[split].get(class_name, 0)
                total_instances_in_split += count
                print(f"  {class_name}: {count} instances")
            
            print(f"  Total instances: {total_instances_in_split}")
            print("-" * 40)
        
        # Print class-wise totals
        print("\nCLASS TOTALS ACROSS ALL SPLITS:")
        grand_total = 0
        for class_name in self.class_names:
            total = total_instances_per_class[class_name]
            grand_total += total
            print(f"  {class_name}: {total} instances")
        
        print(f"  Grand total: {grand_total} instances")
        print("-" * 40)
        
        # Print percentage distribution
        print("\nPERCENTAGE DISTRIBUTION BY CLASS:")
        for class_name in self.class_names:
            total = total_instances_per_class[class_name]
            percentage = (total / grand_total * 100) if grand_total > 0 else 0
            print(f"  {class_name}: {percentage:.2f}%")
    
    def create_visualizations(self):
        """Create visualizations for the dataset statistics."""
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Instance Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        splits_data = []
        classes_data = []
        
        for split in self.splits:
            for class_name in self.class_names:
                count = self.instance_counts[split].get(class_name, 0)
                splits_data.append({
                    'Split': split,
                    'Class': class_name,
                    'Instances': count
                })
        
        df = pd.DataFrame(splits_data)
        
        # 1. Bar plot - Instances per class per split
        ax1 = axes[0, 0]
        pivot_df = df.pivot(index='Class', columns='Split', values='Instances')
        pivot_df.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Instances per Class by Split', fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Instances')
        ax1.legend(title='Split')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for container in ax1.containers:
            ax1.bar_label(container, label_type='edge', fontsize=8)
        
        # 2. Stacked bar plot - Total instances per class
        ax2 = axes[0, 1]
        pivot_df_T = pivot_df.T
        pivot_df_T.plot(kind='bar', stacked=True, ax=ax2, width=0.6)
        ax2.set_title('Total Instances by Split', fontweight='bold')
        ax2.set_xlabel('Split')
        ax2.set_ylabel('Number of Instances')
        ax2.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Pie chart - Overall class distribution
        ax3 = axes[1, 0]
        total_per_class = df.groupby('Class')['Instances'].sum()
        colors = plt.cm.Set3(np.linspace(0, 1, len(total_per_class)))
        wedges, texts, autotexts = ax3.pie(total_per_class.values, 
                                          labels=total_per_class.index,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax3.set_title('Overall Class Distribution', fontweight='bold')
        
        # 4. Heatmap - Split vs Class
        ax4 = axes[1, 1]
        heatmap_data = pivot_df.fillna(0)
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Instance Count Heatmap', fontweight='bold')
        ax4.set_xlabel('Split')
        ax4.set_ylabel('Class')
        
        plt.tight_layout()
        
        # Save the plot
        output_path = 'dataset_instance_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_path}")
        
        plt.show()
    
    def export_to_csv(self):
        """Export the analysis results to CSV format."""
        data = []
        
        for split in self.splits:
            for class_name in self.class_names:
                count = self.instance_counts[split].get(class_name, 0)
                data.append({
                    'Split': split,
                    'Class': class_name,
                    'Instances': count,
                    'Files_in_Split': self.file_counts[split]
                })
        
        df = pd.DataFrame(data)
        csv_path = 'dataset_instance_analysis.csv'
        df.to_csv(csv_path, index=False)
        print(f"Data exported to: {csv_path}")
        
        return df

def main():
    """Main function to run the analysis."""
    try:
        # Initialize analyzer
        analyzer = DatasetAnalyzer()
        
        # Print statistics
        analyzer.print_statistics()
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Export to CSV
        df = analyzer.export_to_csv()
        
        print("\n" + "="*60)
        print("Analysis completed successfully!")
        print("Generated files:")
        print("  - dataset_instance_analysis.png (visualization)")
        print("  - dataset_instance_analysis.csv (raw data)")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
