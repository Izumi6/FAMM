import json
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

def main():
    paper_figures = Path("paper/figures")
    paper_figures.mkdir(parents=True, exist_ok=True)
    
    results_file = Path("experiments/results/large_scale_results.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
            
        if "scalability" in data:
            fig, ax = plt.subplots(figsize=(8, 6))
            scales = sorted([int(k) for k in data["scalability"].keys()])
            systems = list(data["scalability"][str(scales[0])].keys())
            
            for sys in systems:
                ndcg = [data["scalability"][str(s)][sys]["NDCG@10"]["mean"] for s in scales]
                ax.plot(scales, ndcg, marker='o', label=sys)
            
            ax.set_title("Scaling Performance (NDCG@10)")
            ax.set_xlabel("Number of Memories (N)")
            ax.set_ylabel("NDCG@10")
            ax.legend()
            fig.savefig(paper_figures / "fig6_scalability.pdf", bbox_inches='tight')
            print("Generated fig6_scalability.pdf")
            plt.close(fig)
            
        if "ablation_5k" in data:
            fig, ax = plt.subplots(figsize=(8, 6))
            configs = list(data["ablation_5k"].keys())
            ndcg = [data["ablation_5k"][c]["NDCG@10"]["mean"] for c in configs]
            
            ax.bar(configs, ndcg, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_title("Ablation Study (N=5000)")
            ax.set_ylabel("NDCG@10")
            plt.xticks(rotation=45, ha='right')
            fig.savefig(paper_figures / "fig7_ablation_scale.pdf", bbox_inches='tight')
            print("Generated fig7_ablation_scale.pdf")
            plt.close(fig)
            
    # Copy other generated figures to paper/figures
    exp_figures = Path("experiments/figures")
    if exp_figures.exists():
        for f in exp_figures.glob("*.pdf"):
            shutil.copy(f, paper_figures / f.name)
            print(f"Copied {f.name} to paper/figures/")
            
    print("All figures successfully aggregated in paper/figures/")

if __name__ == "__main__":
    main()
