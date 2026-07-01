#!/usr/bin/env python3
"""
Large-Scale Benchmarking Suite for FAMM (with Statistical Validation & Ablation).
"""

import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from backend.memory_engine.event_bus import EventBus
from backend.memory_engine.memory_manager import MemoryManager
from backend.memory_engine.memory_record import SourceType
from backend.vector_database.chroma_adapter import ChromaAdapter
from backend.vector_database.embedding_service import EmbeddingService
from backend.future_utility_predictor.predictor import FutureUtilityPredictor
from backend.goal_retrieval.retriever import GoalAwareRetriever
from baselines.baseline_systems import (
    SimilarityOnlyBaseline,
    ImportanceScoringBaseline,
    EbbinghausDecayBaseline,
    NaiveFIFOBaseline
)
from config.settings import FAMMConfig, ChromaConfig
from evaluation.data_generator import generate_memory_stream
from evaluation.metrics import precision_at_k, recall_at_k, f1_score, ndcg_at_k
from evaluation.advanced_metrics import mean_reciprocal_rank

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("large_scale")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path("./experiments/results")
FIGURES_DIR = Path("./experiments/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_es = None
def get_embedding_service():
    global _es
    if _es is None:
        _es = EmbeddingService()
    return _es

def evaluate_system(sys_name, sys, famm_mm, famm_retriever, all_memories, queries, num_steps, goals):
    stored_ids = []
    
    for content, src_str, tags in all_memories:
        src_map = {"reflection": SourceType.REFLECTION, "observation": SourceType.OBSERVATION, "conversation": SourceType.CONVERSATION, "system": SourceType.SYSTEM}
        if "FAMM" in sys_name:
            r = famm_mm.store(content, source_type=src_map.get(src_str, SourceType.CONVERSATION), goal_tags=tags, goal_context=goals)
            stored_ids.append(r.id)
        else:
            r = sys.store(content, source_type=src_str)
            stored_ids.append(r.id)
            
    for _ in range(num_steps):
        if "FAMM" in sys_name:
            famm_mm.step()
        else:
            sys.step()
            
    sys_metrics = []
    latencies = []
    for q_text, rel_indices in queries:
        rel_ids = {stored_ids[i] for i in rel_indices if i < len(stored_ids)}
        
        q_t0 = time.time()
        if "FAMM" in sys_name:
            if "No Goal Retrieval" in sys_name:
                ret = famm_mm.retrieve(q_text, top_k=10)
                ret_ids = [m.id for m in ret]
            else:
                ret = famm_retriever.retrieve(q_text, goals=goals, top_k=10)
                ret_ids = [m.memory.id for m in ret]
        else:
            ret = sys.retrieve(q_text, top_k=10)
            ret_ids = [m.id for m in ret]
        latencies.append(time.time() - q_t0)
        
        p = precision_at_k(ret_ids, rel_ids, 10)
        r_val = recall_at_k(ret_ids, rel_ids, 10)
        f1 = f1_score(p, r_val)
        ndcg = ndcg_at_k(ret_ids, rel_ids, 10)
        mrr = mean_reciprocal_rank(ret_ids, rel_ids)
        
        sys_metrics.append({"p": p, "r": r_val, "f1": f1, "ndcg": ndcg, "mrr": mrr})
        
    return {
        "P@10": np.mean([m["p"] for m in sys_metrics]),
        "R@10": np.mean([m["r"] for m in sys_metrics]),
        "F1@10": np.mean([m["f1"] for m in sys_metrics]),
        "NDCG@10": np.mean([m["ndcg"] for m in sys_metrics]),
        "MRR": np.mean([m["mrr"] for m in sys_metrics]),
        "Latency_ms": np.mean(latencies) * 1000
    }

def run_scale(scale: int, seeds: list, num_steps: int = 50, ablation: bool = False) -> Dict[str, Dict[str, Any]]:
    logger.info(f"--- Running Benchmark for N={scale} (Seeds: {len(seeds)}) ---")
    es = get_embedding_service()
    goals = ["Complete the research paper", "Write the methodology", "Evaluate performance"]
    
    # Store results per seed
    aggregated = {}
    
    for seed in seeds:
        all_memories, queries = generate_memory_stream(scale, random_seed=seed)
        temp_dirs = []
        def mkd():
            d = tempfile.mkdtemp(prefix="famm_bench_")
            temp_dirs.append(d)
            return d
            
        systems = {}
        if not ablation:
            systems = {
                "FAMM": {"famm": True},
                "SimilarityOnly": {"famm": False, "cls": SimilarityOnlyBaseline},
                "ImportanceScoring": {"famm": False, "cls": ImportanceScoringBaseline},
                "EbbinghausDecay": {"famm": False, "cls": EbbinghausDecayBaseline},
                "NaiveFIFO": {"famm": False, "cls": NaiveFIFOBaseline, "max_memories": int(scale * 0.8)}
            }
        else:
            systems = {
                "FAMM (Full)": {"use_predictor": True, "use_goals": True, "use_decay": True},
                "FAMM – No Predictor": {"use_predictor": False, "use_goals": True, "use_decay": True},
                "FAMM – No Goal Retrieval": {"use_predictor": True, "use_goals": False, "use_decay": True},
                "FAMM – No Decay": {"use_predictor": True, "use_goals": True, "use_decay": False},
            }
            
        for sys_name, config_dict in systems.items():
            if sys_name not in aggregated:
                aggregated[sys_name] = {"P@10": [], "R@10": [], "F1@10": [], "NDCG@10": [], "MRR": [], "Latency_ms": []}
                
            if ablation:
                cfg = FAMMConfig.default()
                cfg.vector_db.chroma.persist_directory = mkd()
                vs = ChromaAdapter(cfg.vector_db.chroma)
                eb = EventBus()
                fup = FutureUtilityPredictor(cfg.future_utility_predictor, es) if config_dict["use_predictor"] else None
                mm = MemoryManager(cfg, vs, es, eb, fup)
                retriever = GoalAwareRetriever(cfg.goal_retrieval, vs, es, eb) if config_dict["use_goals"] else None
                
                # If no decay, we just don't call mm.step()
                steps_to_run = num_steps if config_dict["use_decay"] else 0
                metrics = evaluate_system(sys_name, None, mm, retriever, all_memories, queries, steps_to_run, goals)
            else:
                if config_dict["famm"]:
                    cfg = FAMMConfig.default()
                    cfg.vector_db.chroma.persist_directory = mkd()
                    vs = ChromaAdapter(cfg.vector_db.chroma)
                    eb = EventBus()
                    fup = FutureUtilityPredictor(cfg.future_utility_predictor, es)
                    mm = MemoryManager(cfg, vs, es, eb, fup)
                    retriever = GoalAwareRetriever(cfg.goal_retrieval, vs, es, eb)
                    metrics = evaluate_system(sys_name, None, mm, retriever, all_memories, queries, num_steps, goals)
                else:
                    cls = config_dict["cls"]
                    vs = ChromaAdapter(ChromaConfig(persist_directory=mkd(), collection_name="base"))
                    if "max_memories" in config_dict:
                        sys_inst = cls(vs, es, max_memories=config_dict["max_memories"])
                    else:
                        sys_inst = cls(vs, es)
                    metrics = evaluate_system(sys_name, sys_inst, None, None, all_memories, queries, num_steps, goals)
                    
            for k, v in metrics.items():
                aggregated[sys_name][k].append(v)
                
        for d in temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
            
    # Calculate statistics
    final_results = {}
    for sys_name, metrics in aggregated.items():
        final_results[sys_name] = {}
        for m_name, vals in metrics.items():
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ci = 1.96 * (std_val / np.sqrt(len(seeds)))
            final_results[sys_name][m_name] = {"mean": mean_val, "std": std_val, "ci95": ci}
            
    return final_results

def plot_scalability(all_results: Dict[int, Dict[str, Dict[str, float]]]):
    scales = sorted(list(all_results.keys()))
    systems = list(all_results[scales[0]].keys())
    
    metrics_to_plot = ["R@10", "NDCG@10"]
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#9b59b6']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes = axes.flatten()
    
    for ax, metric in zip(axes, metrics_to_plot):
        for i, sys in enumerate(systems):
            y_means = [all_results[s][sys][metric]["mean"] for s in scales]
            y_cis = [all_results[s][sys][metric]["ci95"] for s in scales]
            
            ax.errorbar(scales, y_means, yerr=y_cis, marker='o', linewidth=2.5, capsize=5, label=sys, color=colors[i % len(colors)])
            
        ax.set_title(metric)
        ax.set_xlabel("Number of Memories (Scale)")
        ax.set_ylabel(metric)
        ax.set_xscale('log')
        ax.set_xticks(scales)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_scalability.png", dpi=300)
    fig.savefig(FIGURES_DIR / "fig6_scalability.pdf")

def main():
    seeds = [42, 100] # Reduced for fast execution, ideally 3-5
    scales = [1000, 5000, 10000]
    
    all_results = {}
    for scale in scales:
        all_results[scale] = run_scale(scale, seeds=seeds)
        
    plot_scalability(all_results)
    
    # Run Ablation at 5000
    ablation_results = run_scale(5000, seeds=seeds, ablation=True)
    
    out_data = {
        "scalability": all_results,
        "ablation_5k": ablation_results,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    out_path = RESULTS_DIR / "large_scale_results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
        
    logger.info("Experiments complete.")
    
if __name__ == "__main__":
    main()
