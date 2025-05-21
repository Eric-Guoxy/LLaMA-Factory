import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_log_file(log_file_path):
    """Parses a single log file to extract step and accuracies."""
    metrics = {}
    step = None
    filename = os.path.basename(log_file_path)

    # Try to extract step from checkpoint name, e.g., "...-checkpoint-100.log"
    match_checkpoint = re.search(r'checkpoint-(\d+)', filename)
    if match_checkpoint:
        step = int(match_checkpoint.group(1))
    elif 'final-sft' in filename: # Placeholder for final model, step will be assigned later
        step = "final" 

    metrics['step_raw'] = step # Store raw step identifier

    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            overall_acc_match = re.search(r"accuracy:\s*(\d+\.?\d*)", content)
            if overall_acc_match:
                metrics['Overall_Accuracy'] = float(overall_acc_match.group(1))
            
            # Regex to find lines like "BENCHMARK_NAME: 0.1234"
            # Exclude lines that are just "accuracy: ..." to avoid double counting overall
            benchmark_matches = re.findall(r"^([\w.-]+):\s*(\d+\.?\d*)$", content, re.MULTILINE)
            for bm_name, bm_acc in benchmark_matches:
                if bm_name.lower() != 'accuracy': # Avoid re-capturing the overall accuracy if it's also in this format
                     metrics[f'{bm_name}_Accuracy'] = float(bm_acc)
    except Exception as e:
        print(f"Error parsing file {log_file_path}: {e}")
    return metrics

def main(logs_directory, output_plot_dir_base):
    all_metrics_data = []
    if not os.path.isdir(logs_directory):
        print(f"Error: Logs directory '{logs_directory}' not found.")
        return

    log_files = [f for f in os.listdir(logs_directory) if f.endswith(".log")]
    if not log_files:
        print(f"No .log files found in '{logs_directory}'.")
        return

    max_step_from_checkpoints = 0
    for log_file in log_files:
        if 'checkpoint-' in log_file:
            match = re.search(r'checkpoint-(\d+)', log_file)
            if match:
                max_step_from_checkpoints = max(max_step_from_checkpoints, int(match.group(1)))

    for log_file in log_files:
        log_file_path = os.path.join(logs_directory, log_file)
        metrics = parse_log_file(log_file_path)
        
        current_step_raw = metrics.get('step_raw')
        if current_step_raw == "final":
            if max_step_from_checkpoints > 0:
                # Assign a step slightly after the last checkpoint for plotting 'final-sft'
                metrics['step'] = max_step_from_checkpoints + (max_step_from_checkpoints // 20 or 5) # Heuristic
            else: # If no checkpoints, maybe assign 0 or a small number if it's the only point
                metrics['step'] = 0 
        elif isinstance(current_step_raw, int):
            metrics['step'] = current_step_raw
        else:
            continue # Skip if step cannot be determined

        if 'step' in metrics:
            all_metrics_data.append(metrics)

    if not all_metrics_data:
        print("No metrics data with valid steps extracted. Exiting visualization.")
        return

    df = pd.DataFrame(all_metrics_data)
    df = df.sort_values(by='step').reset_index(drop=True)
    
    if df.empty or 'step' not in df.columns:
        print("DataFrame is empty or 'step' column is missing after processing. Exiting visualization.")
        return

    # Ensure output directory for plots exists
    output_plot_dir = os.path.join(output_plot_dir_base, "plots")
    os.makedirs(output_plot_dir, exist_ok=True)
    print(f"Saving plots to: {output_plot_dir}")

    metric_columns = [col for col in df.columns if col.endswith('_Accuracy')]

    for metric_col in metric_columns:
        plot_df = df[['step', metric_col]].dropna()
        if not plot_df.empty and len(plot_df) > 1: # Need at least 2 points to draw a line
            plt.figure(figsize=(12, 7))
            plt.plot(plot_df['step'], plot_df[metric_col], marker='o', linestyle='-')
            
            # Add text labels for each point
            for i, point in plot_df.iterrows():
                plt.text(point['step'], point[metric_col], f"{point[metric_col]:.3f}", fontsize=9, ha='left', va='bottom')

            title_name = metric_col.replace('_Accuracy', '').replace('_', ' ')
            plt.title(f'{title_name} vs. Training Steps', fontsize=16)
            plt.xlabel('Training Step', fontsize=14)
            plt.ylabel(title_name, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plot_filename = os.path.join(output_plot_dir, f'{metric_col}_vs_steps.png')
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved plot: {plot_filename}")
        elif not plot_df.empty and len(plot_df) == 1:
            print(f"Only one data point for {metric_col}. Scatter plot not generated, but data point is: Step {plot_df['step'].iloc[0]}, Value {plot_df[metric_col].iloc[0]}")
        else:
            print(f"No data or insufficient data to plot for {metric_col} after dropping NaNs.")
            
    print(f"All plot generation attempts finished. Check {output_plot_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model accuracy from log files.")
    parser.add_argument("logs_directory", type=str, help="Directory containing the .log files.")
    parser.add_argument("--output_plot_dir", type=str, default=".", help="Base directory where the 'plots' subdirectory will be created.")
    args = parser.parse_args()
    main(args.logs_directory, args.output_plot_dir)