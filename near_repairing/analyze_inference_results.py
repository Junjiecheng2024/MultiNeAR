#!/usr/bin/env python3
"""
Analyze NeAR inference results and generate reports.

This script analyzes batch inference results from NeAR model, computing
per-class statistics, generating visualizations, and creating a summary report.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Set plotting style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_results(csv_path, output_dir):
    """Analyze inference results and generate report.
    
    Args:
        csv_path: Path to evaluation results CSV
        output_dir: Output directory for analysis results
    
    Returns:
        stats_df: DataFrame with per-class statistics
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("NeAR Inference Results Analysis")
    print("=" * 80)
    print(f"\nTotal samples: {len(df)}")
    
    # Class names
    class_names = {
        0: "Background",
        1: "Myocardium",
        2: "LA",
        3: "LV",
        4: "RA",
        5: "RV",
        6: "Aorta",
        7: "PA",
        8: "LAA",
        9: "Coronary",
        10: "PV"
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Compute per-class statistics
    stats_list = []
    
    for class_id in range(11):
        col_name = f"dice_class_{class_id}"
        if col_name not in df.columns:
            continue
        
        values = df[col_name].values
        
        stats = {
            'Class_ID': class_id,
            'Class_Name': class_names[class_id],
            'Mean': np.mean(values),
            'Std': np.std(values),
            'Median': np.median(values),
            'Min': np.min(values),
            'Max': np.max(values),
            'Q25': np.percentile(values, 25),
            'Q75': np.percentile(values, 75)
        }
        stats_list.append(stats)
    
    stats_df = pd.DataFrame(stats_list)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Per-Class Dice Statistics")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    
    # Save statistics to CSV
    stats_csv_path = output_path / "class_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\nStatistics saved to: {stats_csv_path}")
    
    # Generate visualizations
    generate_visualizations(df, stats_df, class_names, output_path)
    
    # Generate markdown report
    generate_markdown_report(stats_df, class_names, df, output_path)
    
    return stats_df

def generate_visualizations(df, stats_df, class_names, output_path):
    """Generate visualization plots.
    
    Args:
        df: DataFrame with all results
        stats_df: DataFrame with statistics
        class_names: Dictionary mapping class IDs to names
        output_path: Output directory path
    """
    # 1. Per-class Dice bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(stats_df))
    means = stats_df['Mean'].values
    stds = stats_df['Std'].values
    names = stats_df['Class_Name'].values
    
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                  color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Cardiac Structure', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Per-Class Dice Scores (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "per_class_dice.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'per_class_dice.png'}")
    
    # 2. LAA Dice distribution
    if 'dice_class_8' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        laa_values = df['dice_class_8'].values
        ax.hist(laa_values, bins=30, alpha=0.7, color='coral', edgecolor='black')
        
        mean_laa = np.mean(laa_values)
        ax.axvline(mean_laa, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_laa:.3f}')
        
        ax.set_xlabel('Dice Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('LAA (Left Atrial Appendage) Dice Distribution', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "laa_dice_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path / 'laa_dice_distribution.png'}")
    
    # 3. Box plot for all classes
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dice_cols = [f"dice_class_{i}" for i in range(11) if f"dice_class_{i}" in df.columns]
    dice_data = [df[col].values for col in dice_cols]
    labels = [class_names[int(col.split('_')[-1])] for col in dice_cols]
    
    bp = ax.boxplot(dice_data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Cardiac Structure', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Dice Score Distribution Across All Classes', 
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "dice_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'dice_boxplot.png'}")

def generate_markdown_report(stats_df, class_names, df, output_path):
    """Generate markdown analysis report.
    
    Args:
        stats_df: DataFrame with statistics
        class_names: Dictionary mapping class IDs to names
        df: DataFrame with all results
        output_path: Output directory path
    """
    report_path = output_path / "INFERENCE_REPORT.md"
    
    with open(report_path, 'w') as f:
        f.write("# NeAR Inference Results Analysis Report\n\n")
        
        # Overall summary
        f.write("## Overall Summary\n\n")
        f.write(f"- **Total Samples**: {len(df)}\n")
        
        if 'dice_mean' in df.columns:
            overall_mean = df['dice_mean'].mean()
            overall_std = df['dice_mean'].std()
            f.write(f"- **Overall Mean Dice**: {overall_mean:.4f} ± {overall_std:.4f}\n")
        
        # Exclude background for foreground mean
        fg_cols = [f"dice_class_{i}" for i in range(1, 11) if f"dice_class_{i}" in df.columns]
        if fg_cols:
            fg_mean = df[fg_cols].mean().mean()
            f.write(f"- **Foreground Mean Dice** (Classes 1-10): {fg_mean:.4f}\n\n")
        
        # Per-class statistics table
        f.write("## Per-Class Statistics\n\n")
        f.write("| Class | Name | Mean | Std | Median | Min | Max |\n")
        f.write("|-------|------|------|-----|--------|-----|-----|\n")
        
        for _, row in stats_df.iterrows():
            f.write(f"| {row['Class_ID']} | {row['Class_Name']} | "
                   f"{row['Mean']:.4f} | {row['Std']:.4f} | {row['Median']:.4f} | "
                   f"{row['Min']:.4f} | {row['Max']:.4f} |\n")
        
        # LAA analysis
        if 'dice_class_8' in df.columns:
            f.write("\n## LAA (Left Atrial Appendage) Analysis\n\n")
            laa_values = df['dice_class_8'].values
            laa_mean = np.mean(laa_values)
            laa_std = np.std(laa_values)
            
            f.write(f"- **Mean Dice**: {laa_mean:.4f} ± {laa_std:.4f}\n")
            f.write(f"- **Samples with Dice > 0.5**: {np.sum(laa_values > 0.5)} "
                   f"({np.sum(laa_values > 0.5) / len(laa_values) * 100:.1f}%)\n")
            f.write(f"- **Samples with Dice > 0.6**: {np.sum(laa_values > 0.6)} "
                   f"({np.sum(laa_values > 0.6) / len(laa_values) * 100:.1f}%)\n")
            f.write(f"- **Samples with Dice > 0.7**: {np.sum(laa_values > 0.7)} "
                   f"({np.sum(laa_values > 0.7) / len(laa_values) * 100:.1f}%)\n\n")
        
        # Visualization files
        f.write("## Generated Visualizations\n\n")
        f.write("1. `per_class_dice.png` - Bar plot of mean Dice scores per class\n")
        f.write("2. `laa_dice_distribution.png` - Histogram of LAA Dice scores\n")
        f.write("3. `dice_boxplot.png` - Box plots showing distribution for all classes\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        best_class = stats_df.loc[stats_df['Mean'].idxmax()]
        worst_class = stats_df.loc[stats_df['Mean'].idxmin()]
        
        f.write(f"- **Best performing class**: {best_class['Class_Name']} "
               f"(Dice: {best_class['Mean']:.4f})\n")
        f.write(f"- **Worst performing class**: {worst_class['Class_Name']} "
               f"(Dice: {worst_class['Mean']:.4f})\n")
        
        # Most variable class
        most_variable = stats_df.loc[stats_df['Std'].idxmax()]
        f.write(f"- **Most variable class**: {most_variable['Class_Name']} "
               f"(Std: {most_variable['Std']:.4f})\n\n")
    
    print(f"\nMarkdown report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze NeAR inference results')
    parser.add_argument('--csv', type=str, 
                       default='inference_results_fixed_weights/evaluation_results.csv',
                       help='Path to evaluation results CSV')
    parser.add_argument('--output', type=str,
                       default='inference_results_fixed_weights/analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print(f"Reading results from: {args.csv}")
    print(f"Output directory: {args.output}\n")
    
    stats_df = analyze_results(args.csv, args.output)
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)
    print(f"\nGenerated files in {args.output}:")
    print("  - class_statistics.csv")
    print("  - per_class_dice.png")
    print("  - laa_dice_distribution.png")
    print("  - dice_boxplot.png")
    print("  - INFERENCE_REPORT.md")

if __name__ == "__main__":
    main()
