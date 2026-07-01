#!/usr/bin/env python3
"""
Model Search — Train and compare ML models for Future Utility Prediction.
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("model_search")

FIGURES_DIR = Path("./experiments/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def generate_training_data(n_samples: int = 5000):
    """Generate synthetic retrospective feature data."""
    np.random.seed(42)
    # Features: [sim, recency, freq, src_prior, entity_overlap]
    X = np.random.rand(n_samples, 5)
    
    # Base utility based on a linear combination of features
    # Similar to the heuristic weights: [0.35, 0.20, 0.15, 0.15, 0.15]
    weights = np.array([0.35, 0.20, 0.15, 0.15, 0.15])
    y_base = np.dot(X, weights)
    
    # Add non-linear interactions to make ML models useful
    y_interaction = 0.1 * (X[:, 0] * X[:, 1]) + 0.1 * (X[:, 2] ** 2)
    
    # Add noise
    noise = np.random.normal(0, 0.05, n_samples)
    
    y = np.clip(y_base + y_interaction + noise, 0.0, 1.0)
    
    return X, y

def run_model_search():
    logger.info("Generating dataset...")
    X, y = generate_training_data(n_samples=5000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Small MLP": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=100, random_state=42),
        "Linear (Ridge)": Ridge(alpha=1.0)
    }
    
    results = {}
    
    logger.info(f"{'Model':<20} | {'CV MSE (Train)':<15} | {'Test MSE':<10} | {'Latency (ms)':<12}")
    logger.info("-" * 65)
    
    for name, model in models.items():
        # CV on Train
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -np.mean(cv_scores)
        
        # Train full and eval
        model.fit(X_train, y_train)
        
        t0 = time.time()
        y_pred = model.predict(X_test)
        latency = (time.time() - t0) / len(X_test) * 1000 # ms per sample
        
        test_mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {
            "cv_mse": cv_mse,
            "test_mse": test_mse,
            "latency": latency
        }
        
        logger.info(f"{name:<20} | {cv_mse:<15.6f} | {test_mse:<10.6f} | {latency:<12.4f}")
        
    # Plotting
    names = list(results.keys())
    mses = [results[n]["test_mse"] for n in names]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, mses, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')
                    
    ax.set_ylabel('Test MSE (Lower is better)')
    ax.set_title('Future Utility Predictor: Model Comparison')
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig7_model_search.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig7_model_search.pdf")
    logger.info(f"Saved plot to {FIGURES_DIR}/fig7_model_search.png")
    
if __name__ == "__main__":
    run_model_search()
