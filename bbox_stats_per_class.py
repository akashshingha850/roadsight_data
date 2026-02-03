import os
import numpy as np
from glob import glob

def compute_bbox_stats(annotation_pattern, classes):
    """
    Compute bounding box statistics for each class from YOLO format annotations.
    
    Args:
        annotation_pattern (str): Glob pattern for annotation files.
        classes (dict): Dictionary mapping class id to class name.
    
    Returns:
        dict: Statistics for each class.
    """
    stats = {class_name: {'widths': [], 'heights': [], 'areas': [], 'aspect_ratios': []} for class_name in classes.values()}
    
    # Find all annotation files
    annotation_files = glob(annotation_pattern)
    
    for file in annotation_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(float(parts[0]))
                    if class_id in classes:
                        class_name = classes[class_id]
                        x_center, y_center, width, height = map(float, parts[1:])
                        area = width * height
                        aspect_ratio = width / height if height != 0 else 0
                        stats[class_name]['widths'].append(width)
                        stats[class_name]['heights'].append(height)
                        stats[class_name]['areas'].append(area)
                        stats[class_name]['aspect_ratios'].append(aspect_ratio)
    
    # Compute statistics
    results = {}
    for class_name, data in stats.items():
        if data['widths']:
            results[class_name] = {
                'width': {
                    'mean_std': f"{np.mean(data['widths']):.3f} ± {np.std(data['widths']):.3f}",
                    'median': f"{np.median(data['widths']):.3f}",
                    'range': f"{np.min(data['widths']):.3f} - {np.max(data['widths']):.3f}"
                },
                'height': {
                    'mean_std': f"{np.mean(data['heights']):.3f} ± {np.std(data['heights']):.3f}",
                    'median': f"{np.median(data['heights']):.3f}",
                    'range': f"{np.min(data['heights']):.3f} - {np.max(data['heights']):.3f}"
                },
                'area': {
                    'mean_std': f"{np.mean(data['areas']):.3f} ± {np.std(data['areas']):.3f}",
                    'median': f"{np.median(data['areas']):.3f}",
                    'range': f"{np.min(data['areas']):.3f} - {np.max(data['areas']):.3f}"
                },
                'aspect_ratio': {
                    'mean_std': f"{np.mean(data['aspect_ratios']):.3f} ± {np.std(data['aspect_ratios']):.3f}",
                    'median': f"{np.median(data['aspect_ratios']):.3f}",
                    'range': f"{np.min(data['aspect_ratios']):.3f} - {np.max(data['aspect_ratios']):.3f}"
                }
            }
        else:
            results[class_name] = {metric: {'mean_std': 'N/A', 'median': 'N/A', 'range': 'N/A'} for metric in ['width', 'height', 'area', 'aspect_ratio']}
    
    return results

def generate_latex_table(stats):
    """
    Generate LaTeX table for all classes.
    
    Args:
        stats (dict): Statistics for each class.
    
    Returns:
        str: LaTeX table string.
    """
    class_names = list(stats.keys())
    table = f"""\\begin{{table}}[h]
\\caption{{Bounding box statistics for each class showing width, height, area, and aspect ratio distributions with normalized coordinates (0-1 range)}}
\\label{{tab:bbox-stats-classes}}
\\begin{{center}}
\\begin{{tabular}}{{l{'c' * (len(class_names) * 3)}}}
\\multicolumn{{1}}{{c}}{{\\bf Metric}} & \\multicolumn{{3}}{{c}}{{\\bf {class_names[0]}}} & \\multicolumn{{3}}{{c}}{{\\bf {class_names[1]}}}
\\\\ & Mean ± Std & Median & Range & Mean ± Std & Median & Range
\\\\ \\hline \\\\
"""
    metrics = ['Width', 'Height', 'Area', 'Aspect Ratio']
    for metric in metrics:
        row = f"{metric}"
        for class_name in class_names:
            row += f" & {stats[class_name][metric.lower().replace(' ', '_')]['mean_std']} & {stats[class_name][metric.lower().replace(' ', '_')]['median']} & {stats[class_name][metric.lower().replace(' ', '_')]['range']}"
        row += " \\\\\n"
        table += row
    table += """\\end{tabular}
\\end{center}
\\end{table}
"""
    return table

if __name__ == "__main__":
    # Define classes
    classes = {0: 'roundabout', 1: 'intersection'}
    
    # Glob pattern for all annotation files
    annotation_pattern = 'data/*/labels/*.txt'
    
    stats = compute_bbox_stats(annotation_pattern, classes)
    
    print(generate_latex_table(stats))