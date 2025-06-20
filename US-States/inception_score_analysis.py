import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd

def load_pickle(file_path):
    """Load pickle file containing training results"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_inception_scores(pickle_data, output_dir):
    """Analyze and visualize inception scores from training"""
    if 'inception_scores' not in pickle_data:
        print("No inception scores found in the pickle file")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    inception_scores = pickle_data['inception_scores']
    tasks = sorted(inception_scores.keys())
    
    # 1. Overall inception score progression across all tasks
    plt.figure(figsize=(12, 8))
    
    for task in tasks:
        rounds = sorted(inception_scores[task].keys())
        scores = [inception_scores[task][r]['overall_score'] for r in rounds]
        stds = [inception_scores[task][r]['overall_std'] for r in rounds]
        
        # Adjust rounds to be continuous across tasks
        if task > 0:
            prev_task_rounds = sorted(inception_scores[task-1].keys())
            offset = max(prev_task_rounds) + 1
            plot_rounds = [r + offset for r in rounds]
        else:
            plot_rounds = rounds
            
        plt.errorbar(plot_rounds, scores, yerr=stds, marker='o', linestyle='-', label=f'Task {task}')
    
    plt.title('Inception Score Progression Across All Tasks')
    plt.xlabel('Training Round')
    plt.ylabel('Inception Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_inception_scores.png'))
    plt.close()
    
    # 2. Final inception score for each task
    final_scores = []
    final_stds = []
    
    for task in tasks:
        rounds = sorted(inception_scores[task].keys())
        if rounds:
            final_scores.append(inception_scores[task][rounds[-1]]['overall_score'])
            final_stds.append(inception_scores[task][rounds[-1]]['overall_std'])
        else:
            final_scores.append(0)
            final_stds.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.bar(tasks, final_scores, yerr=final_stds, capsize=5)
    plt.title('Final Inception Score for Each Task')
    plt.xlabel('Task')
    plt.ylabel('Inception Score')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_task_scores.png'))
    plt.close()
    
    # 3. Class-specific inception scores
    # For each task, find the final round
    for task in tasks:
        rounds = sorted(inception_scores[task].keys())
        if not rounds:
            continue
            
        final_round = rounds[-1]
        
        if 'class_scores' not in inception_scores[task][final_round]:
            continue
            
        class_scores = inception_scores[task][final_round]['class_scores']
        classes = sorted(class_scores.keys())
        scores = [class_scores[c][0] for c in classes]
        stds = [class_scores[c][1] for c in classes]
        
        plt.figure(figsize=(12, 6))
        plt.bar(classes, scores, yerr=stds, capsize=5)
        plt.title(f'Class-Specific Inception Scores for Task {task}')
        plt.xlabel('Class')
        plt.ylabel('Inception Score')
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_scores_task_{task}.png'))
        plt.close()
    
    # 4. Create a heatmap of class scores across tasks
    # Collect all classes across all tasks
    all_classes = set()
    for task in tasks:
        for round_key in inception_scores[task]:
            if 'class_scores' in inception_scores[task][round_key]:
                all_classes.update(inception_scores[task][round_key]['class_scores'].keys())
    
    all_classes = sorted(list(all_classes))
    heatmap_data = np.zeros((len(tasks), len(all_classes)))
    
    for i, task in enumerate(tasks):
        rounds = sorted(inception_scores[task].keys())
        if not rounds:
            continue
            
        final_round = rounds[-1]
        
        if 'class_scores' not in inception_scores[task][final_round]:
            continue
            
        class_scores = inception_scores[task][final_round]['class_scores']
        
        for j, class_label in enumerate(all_classes):
            if class_label in class_scores:
                heatmap_data[i, j] = class_scores[class_label][0]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", xticklabels=all_classes, yticklabels=tasks, cmap="YlGnBu")
    plt.title('Class-Specific Inception Scores Across Tasks')
    plt.xlabel('Class')
    plt.ylabel('Task')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_scores_heatmap.png'))
    plt.close()
    
    # 5. Generate comprehensive report
    report_path = os.path.join(output_dir, 'inception_score_report.md')
    with open(report_path, 'w') as f:
        f.write('# Inception Score Analysis Report\n\n')
        
        f.write('## Overall Inception Score Progression\n\n')
        f.write('![Overall Inception Scores](overall_inception_scores.png)\n\n')
        
        f.write('## Final Inception Score by Task\n\n')
        f.write('![Final Task Scores](final_task_scores.png)\n\n')
        
        f.write('| Task | Inception Score | Standard Deviation |\n')
        f.write('|------|-----------------|--------------------|\n')
        for i, task in enumerate(tasks):
            f.write(f'| {task} | {final_scores[i]:.4f} | {final_stds[i]:.4f} |\n')
        f.write('\n')
        
        f.write('## Class-Specific Inception Scores\n\n')
        f.write('![Class Scores Heatmap](class_scores_heatmap.png)\n\n')
        
        for task in tasks:
            f.write(f'### Task {task}\n\n')
            f.write(f'![Class Scores Task {task}](class_scores_task_{task}.png)\n\n')
            
            rounds = sorted(inception_scores[task].keys())
            if not rounds:
                f.write('No data available for this task.\n\n')
                continue
                
            final_round = rounds[-1]
            
            if 'class_scores' not in inception_scores[task][final_round]:
                f.write('No class-specific scores available for this task.\n\n')
                continue
                
            class_scores = inception_scores[task][final_round]['class_scores']
            classes = sorted(class_scores.keys())
            
            f.write('| Class | Inception Score | Standard Deviation |\n')
            f.write('|-------|-----------------|--------------------|\n')
            for c in classes:
                score, std = class_scores[c]
                f.write(f'| {c} | {score:.4f} | {std:.4f} |\n')
            f.write('\n')

def create_comparison_visualizations(pickle_files, labels, output_dir):
    """Create visualizations comparing multiple runs with different configurations"""
    if len(pickle_files) <= 1:
        print("Need at least two pickle files for comparison")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all pickle files
    pickles = []
    for file_path in pickle_files:
        with open(file_path, 'rb') as f:
            pickles.append(pickle.load(f))
    
    # Check if all pickles have inception scores
    for i, p in enumerate(pickles):
        if 'inception_scores' not in p:
            print(f"No inception scores found in pickle file {pickle_files[i]}")
            return
    
    # 1. Compare final inception scores across configurations
    plt.figure(figsize=(12, 8))
    
    for i, (p, label) in enumerate(zip(pickles, labels)):
        inception_scores = p['inception_scores']
        tasks = sorted(inception_scores.keys())
        
        final_scores = []
        final_stds = []
        
        for task in tasks:
            rounds = sorted(inception_scores[task].keys())
            if rounds:
                final_scores.append(inception_scores[task][rounds[-1]]['overall_score'])
                final_stds.append(inception_scores[task][rounds[-1]]['overall_std'])
            else:
                final_scores.append(0)
                final_stds.append(0)
        
        x = np.array(tasks) + 0.1 * (i - len(pickles)/2 + 0.5)  # Offset bars
        plt.bar(x, final_scores, width=0.1, yerr=final_stds, capsize=5, label=label)
    
    plt.title('Final Inception Scores Comparison')
    plt.xlabel('Task')
    plt.ylabel('Inception Score')
    plt.xticks(tasks)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_scores_comparison.png'))
    plt.close()
    
    # 2. Create a line plot comparing inception score progression
    # First, determine the maximum number of tasks
    max_tasks = 0
    for p in pickles:
        max_tasks = max(max_tasks, max(p['inception_scores'].keys()) + 1)
    
    # Create a figure with subplots for each task
    fig = plt.figure(figsize=(15, 5 * max_tasks))
    gs = GridSpec(max_tasks, 1, figure=fig)
    
    for task in range(max_tasks):
        ax = fig.add_subplot(gs[task, 0])
        
        for i, (p, label) in enumerate(zip(pickles, labels)):
            inception_scores = p['inception_scores']
            
            if task not in inception_scores:
                continue
                
            rounds = sorted(inception_scores[task].keys())
            scores = [inception_scores[task][r]['overall_score'] for r in rounds]
            stds = [inception_scores[task][r]['overall_std'] for r in rounds]
            
            ax.errorbar(rounds, scores, yerr=stds, marker='o', linestyle='-', label=label)
        
        ax.set_title(f'Inception Score Progression - Task {task}')
        ax.set_xlabel('Training Round')
        ax.set_ylabel('Inception Score')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'progression_comparison.png'))
    plt.close()
    
    # 3. Generate comparison report
    report_path = os.path.join(output_dir, 'comparison_report.md')
    with open(report_path, 'w') as f:
        f.write('# Inception Score Comparison Report\n\n')
        
        f.write('## Final Inception Scores Comparison\n\n')
        f.write('![Final Scores Comparison](final_scores_comparison.png)\n\n')
        
        f.write('## Inception Score Progression Comparison\n\n')
        f.write('![Progression Comparison](progression_comparison.png)\n\n')
        
        f.write('## Detailed Comparison\n\n')
        
        # Create a table with all final scores
        f.write('### Final Inception Scores\n\n')
        f.write('| Configuration | ' + ' | '.join([f'Task {t}' for t in range(max_tasks)]) + ' |\n')
        f.write('|---------------| ' + ' | '.join(['-------' for _ in range(max_tasks)]) + ' |\n')
        
        for i, (p, label) in enumerate(zip(pickles, labels)):
            inception_scores = p['inception_scores']
            row = f'| {label} |'
            
            for task in range(max_tasks):
                if task in inception_scores:
                    rounds = sorted(inception_scores[task].keys())
                    if rounds:
                        final_score = inception_scores[task][rounds[-1]]['overall_score']
                        final_std = inception_scores[task][rounds[-1]]['overall_std']
                        row += f' {final_score:.4f} ± {final_std:.4f} |'
                    else:
                        row += ' N/A |'
                else:
                    row += ' N/A |'
            
            f.write(row + '\n')
        
        f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Analyze inception scores from PreciseFCL training')
    parser.add_argument('--pickle_file', type=str, help='Path to the pickle file containing training results')
    parser.add_argument('--output_dir', type=str, default='inception_score_analysis', help='Directory to save analysis results')
    parser.add_argument('--comparison', action='store_true', help='Enable comparison mode for multiple pickle files')
    parser.add_argument('--pickle_files', nargs='+', help='List of pickle files for comparison')
    parser.add_argument('--labels', nargs='+', help='Labels for each configuration in comparison')
    
    args = parser.parse_args()
    
    if args.comparison:
        if not args.pickle_files or len(args.pickle_files) < 2:
            print("Please provide at least two pickle files for comparison")
            return
            
        if not args.labels or len(args.labels) != len(args.pickle_files):
            print("Please provide labels for each configuration (same number as pickle files)")
            return
            
        create_comparison_visualizations(args.pickle_files, args.labels, args.output_dir)
    else:
        if not args.pickle_file:
            print("Please provide a pickle file")
            return
            
        pickle_data = load_pickle(args.pickle_file)
        analyze_inception_scores(pickle_data, args.output_dir)
    
if __name__ == "__main__":
    main()