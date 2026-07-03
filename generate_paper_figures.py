"""Regenerate all matplotlib figures for the FAMM paper with correct data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

fig_dir = Path("paper/figures")
fig_dir.mkdir(exist_ok=True)

# Load actual results
with open("experiments/results/large_scale_results.json") as f:
    data = json.load(f)

# ─── Fig 1: Decay Curves ───
fig, ax = plt.subplots(figsize=(4.5, 3.2))
steps = np.arange(0, 200)
base, kappa = 0.05, 2.0

for u0, ls, label in [(0.9, '-', 'U₀ = 0.9 (high utility)'), 
                        (0.5, '--', 'U₀ = 0.5 (medium)'),
                        (0.1, ':', 'U₀ = 0.1 (low utility)')]:
    u_vals = [u0]
    for _ in range(len(steps)-1):
        u = u_vals[-1]
        d = base * ((1.0 - u) ** kappa)
        u_vals.append(max(0, u - d))
    ax.plot(steps, u_vals, ls, linewidth=2, label=label)

ax.axhline(y=0.05, color='red', linestyle='-.', alpha=0.5, label='Prune threshold (θ=0.05)')
ax.set_xlabel('Decay Cycles')
ax.set_ylabel('Utility Score U(t)')
ax.set_title('Utility-Conditioned Decay Curves')
ax.legend(loc='center right')
ax.set_ylim(-0.02, 1.02)
fig.savefig(fig_dir / "fig1_decay_curves.pdf")
fig.savefig(fig_dir / "fig1_decay_curves.png")
plt.close()
print("✓ Fig 1: Decay curves")

# ─── Fig 4: Ablation Study ───
ablation = data['ablation_5k']
configs = ['FAMM (Full)', 'FAMM – No Predictor', 'FAMM – No Goal Retrieval', 'FAMM – No Decay']
short_labels = ['Full', '−Predictor', '−Goal Retr.', '−Decay']
metrics_names = ['P@10', 'R@10', 'F1@10', 'NDCG@10']

fig, ax = plt.subplots(figsize=(5, 3.5))
x = np.arange(len(short_labels))
width = 0.18
colors = ['#2c3e50', '#3498db', '#e74c3c', '#2ecc71']

for i, metric in enumerate(metrics_names):
    vals = [ablation[c][metric]['mean'] for c in configs]
    bars = ax.bar(x + i*width - 1.5*width, vals, width, label=metric, color=colors[i], edgecolor='white', linewidth=0.5)

ax.set_ylabel('Score')
ax.set_title('Ablation Study (N = 5,000)')
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=15, ha='right')
ax.legend(loc='upper left', ncol=2)
ax.set_ylim(0, 0.85)
fig.savefig(fig_dir / "fig4_ablation.pdf")
fig.savefig(fig_dir / "fig4_ablation.png")
plt.close()
print("✓ Fig 4: Ablation study")

# ─── Fig 5: Feature Importance (CORRECTED) ───
fig, ax = plt.subplots(figsize=(4.5, 3))
features = ['Goal\nSimilarity', 'Recency', 'Access\nFrequency', 'Source Type\nPrior', 'Entity\nOverlap']
weights = [0.35, 0.20, 0.15, 0.15, 0.15]
colors_feat = ['#2c3e50', '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7']

bars = ax.barh(features, weights, color=colors_feat, edgecolor='white', linewidth=0.5, height=0.6)
ax.set_xlabel('Feature Weight')
ax.set_title('Heuristic Scorer Feature Weights')
ax.set_xlim(0, 0.42)
for bar, w in zip(bars, weights):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{w:.2f}', va='center', fontsize=9)
ax.invert_yaxis()
fig.savefig(fig_dir / "fig5_feature_importance.pdf")
fig.savefig(fig_dir / "fig5_feature_importance.png")
plt.close()
print("✓ Fig 5: Feature importance (CORRECTED)")

# ─── Fig 6: Scalability ───
fig, ax = plt.subplots(figsize=(4.5, 3.2))
scales = [1000, 5000, 10000]
systems = ['FAMM', 'SimilarityOnly', 'ImportanceScoring', 'EbbinghausDecay', 'NaiveFIFO']
labels = ['FAMM (Full)', 'Similarity-Only', 'Importance Scoring', 'Ebbinghaus Decay', 'Naive FIFO']
markers = ['o', 's', '^', 'D', 'v']
colors_sys = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

for sys, label, marker, color in zip(systems, labels, markers, colors_sys):
    ndcg_vals = [data['scalability'][str(n)][sys]['NDCG@10']['mean'] for n in scales]
    ax.plot(scales, ndcg_vals, f'-{marker}', label=label, color=color, linewidth=1.8, markersize=6)

ax.set_xlabel('Memory Database Size (N)')
ax.set_ylabel('NDCG@10')
ax.set_title('Retrieval Quality vs. Scale')
ax.set_xticks(scales)
ax.set_xticklabels(['1K', '5K', '10K'])
ax.legend(loc='lower left', fontsize=8)
ax.set_ylim(0.35, 0.95)
fig.savefig(fig_dir / "fig6_scalability.pdf")
fig.savefig(fig_dir / "fig6_scalability.png")
plt.close()
print("✓ Fig 6: Scalability")

print("\nAll figures regenerated in", fig_dir)
