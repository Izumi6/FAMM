import json
import csv
import os
from pathlib import Path

def generate_csv_results():
    results_dir = Path("experiments/results")
    json_path = results_dir / "large_scale_results.json"
    csv_path_scale = results_dir / "large_scale_results.csv"
    csv_path_ablation = results_dir / "ablation_results.csv"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Scalability CSV
    if "scalability" in data:
        with open(csv_path_scale, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Scale", "System", "Metric", "Mean", "Std", "CI95"])
            for scale, systems in data["scalability"].items():
                for system, metrics in systems.items():
                    for metric, values in metrics.items():
                        writer.writerow([scale, system, metric, values["mean"], values["std"], values["ci95"]])
        print(f"Generated {csv_path_scale}")
                        
    # Ablation CSV
    if "ablation_5k" in data:
        with open(csv_path_ablation, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Configuration", "Metric", "Mean", "Std", "CI95"])
            for config, metrics in data["ablation_5k"].items():
                for metric, values in metrics.items():
                    writer.writerow([config, metric, values["mean"], values["std"], values["ci95"]])
        print(f"Generated {csv_path_ablation}")

if __name__ == "__main__":
    generate_csv_results()
