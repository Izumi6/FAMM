#!/usr/bin/env python3
"""
run_full_experiments.py — Complete FAMM experiment suite.

Runs:
1. Synthetic scenario (small, controlled)
2. Scaled scenario (100+ memories)  
3. Ablation studies (disable each module)
4. Generates all paper figures
"""

import json
import logging
import math
import shutil
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from backend.consolidation.consolidator import Consolidator
from backend.forgetting_engine.utility_decay import UtilityDecay
from backend.forgetting_engine.ebbinghaus_baseline import EbbinghausBaseline
from backend.future_utility_predictor.predictor import FutureUtilityPredictor
from backend.goal_retrieval.retriever import GoalAwareRetriever
from backend.memory_engine.event_bus import EventBus
from backend.memory_engine.memory_manager import MemoryManager
from backend.memory_engine.memory_record import MemoryRecord, MemoryState, SourceType
from backend.vector_database.chroma_adapter import ChromaAdapter
from backend.vector_database.embedding_service import EmbeddingService
from baselines.baseline_systems import (
    EbbinghausDecayBaseline,
    ImportanceScoringBaseline,
    NaiveFIFOBaseline,
    SimilarityOnlyBaseline,
)
from config.settings import ChromaConfig, FAMMConfig, ForgettingEngineConfig
from evaluation.metrics import compute_all_metrics, precision_at_k, recall_at_k, f1_score, ndcg_at_k

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("experiment")
logger.setLevel(logging.INFO)

FIGURES_DIR = Path("./experiments/figures")
RESULTS_DIR = Path("./experiments/results")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared resources ──
_embedding_service: EmbeddingService | None = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        _ = _embedding_service.dimension  # force load
    return _embedding_service


# ═══════════════════════════════════════════════════════════
# FIGURE 1: Decay Curve Comparison (FAMM vs Ebbinghaus)
# ═══════════════════════════════════════════════════════════

def generate_decay_curves():
    """Generate paper Figure 1: Adaptive vs uniform decay curves."""
    logger.info("Generating Figure 1: Decay curves...")

    famm_decay = UtilityDecay(ForgettingEngineConfig())
    ebb_decay = EbbinghausBaseline(stability=24.0)

    steps = 100

    # FAMM curves at different initial utilities
    famm_high = famm_decay.project_decay_timeline(initial_utility=0.9, steps=steps)
    famm_mid = famm_decay.project_decay_timeline(initial_utility=0.5, steps=steps)
    famm_low = famm_decay.project_decay_timeline(initial_utility=0.2, steps=steps)

    # Ebbinghaus (uniform for all)
    ebb_curve = ebb_decay.project_decay_timeline(steps=steps, hours_per_step=1.0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = range(steps + 1)

    ax.plot(x, famm_high, 'b-', linewidth=2.5, label='FAMM (utility=0.9)', alpha=0.9)
    ax.plot(x, famm_mid, 'g-', linewidth=2.5, label='FAMM (utility=0.5)', alpha=0.9)
    ax.plot(x, famm_low, 'orange', linewidth=2.5, label='FAMM (utility=0.2)', alpha=0.9)
    ax.plot(x, ebb_curve, 'r--', linewidth=2.5, label='Ebbinghaus (uniform)', alpha=0.8)

    ax.axhline(y=0.05, color='gray', linestyle=':', alpha=0.5, label='Prune threshold')
    ax.set_xlabel('Decay Steps', fontsize=13)
    ax.set_ylabel('Utility / Retention Score', fontsize=13)
    ax.set_title('Adaptive Utility-Conditioned Decay vs Uniform Ebbinghaus Decay', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    sns.despine()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_decay_curves.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig1_decay_curves.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("  → Saved fig1_decay_curves.png/pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Utility Score Distribution by Source Type
# ═══════════════════════════════════════════════════════════

def generate_utility_distribution():
    """Generate paper Figure 2: Utility score distribution."""
    logger.info("Generating Figure 2: Utility distribution...")

    es = get_embedding_service()
    config = FAMMConfig.default()
    predictor = FutureUtilityPredictor(config.future_utility_predictor, es)

    goals = [
        "Complete the research paper on memory management",
        "Run experiments comparing FAMM against baselines",
    ]

    # Generate memories of each source type
    source_memories = {
        "Reflection": [
            "Error rates increase when memory exceeds 5000 records, suggesting we need better pruning.",
            "The utility-conditioned decay formula should use an exponent of 2.0 for optimal performance.",
            "Our hypothesis about future-aware scoring is supported by the initial experiment results.",
            "The consolidation module reduces memory count by 30% while preserving retrieval quality.",
        ],
        "Observation": [
            "The experiment completed successfully with precision@10 of 0.85.",
            "ChromaDB query latency averages 2.3ms for 10K records.",
            "The agent correctly answered 78% of multi-hop reasoning questions.",
            "FAISS index build time scales linearly with record count.",
        ],
        "Conversation": [
            "The user asked about the weather forecast for tomorrow.",
            "Can you remind me to buy groceries after work today?",
            "I prefer dark mode for all my applications and tools.",
            "What time is the team meeting scheduled for this week?",
        ],
        "System": [
            "ChromaDB uses HNSW indexing for approximate nearest neighbor search.",
            "Python scikit-learn supports MLPRegressor for small neural networks.",
            "The FAISS library provides GPU-accelerated similarity search.",
            "Pydantic v2 uses Rust-based core for faster validation.",
        ],
    }

    source_map = {
        "Reflection": SourceType.REFLECTION,
        "Observation": SourceType.OBSERVATION,
        "Conversation": SourceType.CONVERSATION,
        "System": SourceType.SYSTEM,
    }

    data = {src: [] for src in source_memories}

    for src_name, contents in source_memories.items():
        for content in contents:
            record = MemoryRecord(
                content=content,
                source_type=source_map[src_name],
            )
            record.embedding = es.encode(content)
            score = predictor.predict(record, goal_context=goals)
            data[src_name].append(score)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    positions = list(range(len(data)))
    
    bp = ax.boxplot(
        [data[k] for k in data],
        tick_labels=list(data.keys()),
        patch_artist=True,
        widths=0.5,
    )
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (src, scores) in enumerate(data.items()):
        jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(scores))
        ax.scatter([i + 1 + j for j in jitter], scores, color=colors[i], alpha=0.8, s=60, zorder=5, edgecolors='white', linewidth=0.5)

    ax.set_ylabel('Predicted Future Utility Score', fontsize=13)
    ax.set_title('Utility Score Distribution by Memory Source Type', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_utility_distribution.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig2_utility_distribution.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("  → Saved fig2_utility_distribution.png/pdf")


# ═══════════════════════════════════════════════════════════
# SCALED EXPERIMENT (100+ memories)
# ═══════════════════════════════════════════════════════════

def create_scaled_scenario():
    """Create a large-scale scenario with 100 memories."""
    
    # Research-relevant memories (indices 0-24)
    research_memories = [
        ("Transformer attention mechanisms use O(n²) memory for sequence length n.", "reflection", ["research"]),
        ("The FAMM framework introduces future-aware utility prediction for memory management.", "reflection", ["research", "writing"]),
        ("MemGPT uses a two-tier memory architecture inspired by operating system virtual memory.", "reflection", ["research", "literature"]),
        ("Retrieval-augmented generation improves factual accuracy by grounding in external knowledge.", "reflection", ["research"]),
        ("Long-context LLMs like Gemini 1.5 support up to 1M tokens but still benefit from memory systems.", "reflection", ["research"]),
        ("The forgetting curve proposed by Ebbinghaus follows an exponential decay pattern.", "reflection", ["research", "methodology"]),
        ("Memory consolidation in neuroscience involves hippocampal replay during sleep.", "system", ["research"]),
        ("Goal-conditioned retrieval outperforms similarity-only retrieval by 15% on multi-hop tasks.", "observation", ["research", "evaluation"]),
        ("The utility-conditioned decay formula achieves 80x speed difference between high and low utility.", "observation", ["research", "evaluation"]),
        ("Ablation studies show the Future Utility Predictor contributes 40% of FAMM's improvement.", "observation", ["research", "evaluation"]),
        ("The multi-signal ranker uses four weighted components for goal-aware scoring.", "reflection", ["research", "methodology"]),
        ("ChromaDB provides native metadata filtering which is essential for state-based queries.", "system", ["implementation"]),
        ("FAISS IndexFlatIP with normalized vectors is equivalent to cosine similarity search.", "system", ["implementation"]),
        ("The evaluation uses Precision@K, Recall@K, F1, and NDCG as primary metrics.", "reflection", ["evaluation"]),
        ("LoCoMo benchmark contains 50 long-term conversational memory evaluation sessions.", "system", ["evaluation"]),
        ("Sentence Transformers all-MiniLM-L6-v2 produces 384-dimensional normalized embeddings.", "system", ["implementation"]),
        ("The adaptive decay exponent of 2.0 provides optimal separation between utility classes.", "observation", ["research", "methodology"]),
        ("Memory consolidation reduces storage by 30% while preserving 95% retrieval quality.", "observation", ["research", "evaluation"]),
        ("The event bus architecture enables clean ablation by disconnecting module handlers.", "reflection", ["implementation"]),
        ("Retrospective labeling creates training data by tracking which memories were actually used.", "reflection", ["research", "methodology"]),
        ("The learned scorer MLP achieves 0.003 MSE after training on 500 retrospective samples.", "observation", ["research", "evaluation"]),
        ("Goal alignment scoring uses max aggregation over all active goals for each memory.", "reflection", ["methodology"]),
        ("The pruning threshold of 0.05 balances storage efficiency with information preservation.", "observation", ["research"]),
        ("Pydantic models ensure type-safe configuration across all FAMM modules.", "system", ["implementation"]),
        ("The lifecycle state machine supports reactivation: STALE → ACTIVE on successful retrieval.", "reflection", ["research", "methodology"]),
    ]

    # Irrelevant noise memories (indices 25-74) 
    noise_memories = []
    noise_topics = [
        ("The weather forecast predicts {} for the coming {}.", "conversation", ["personal"]),
        ("Remember to {} after work on {}.", "conversation", ["personal"]),
        ("The {} restaurant on {} street has great reviews.", "conversation", ["dining"]),
        ("Flight {} departs at {} from terminal {}.", "conversation", ["travel"]),
        ("The {} meeting was rescheduled to {} PM.", "conversation", ["scheduling"]),
    ]
    
    weather = ["rain", "sunshine", "clouds", "snow", "storms", "clear skies", "fog", "wind", "hail", "drizzle"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "tomorrow", "next week", "the weekend"]
    tasks = ["buy groceries", "pick up laundry", "call the dentist", "renew insurance", "fix the bike", "walk the dog", "water plants", "pay bills", "clean the house", "update resume"]
    restaurants = ["Italian", "Japanese", "Mexican", "Thai", "Indian", "French", "Korean", "Greek", "Lebanese", "Vietnamese"]
    streets = ["Main", "Oak", "Elm", "Pine", "Maple", "Cedar", "Birch", "Walnut", "Cherry", "Spruce"]
    flights = ["AA123", "UA456", "DL789", "SW012", "BA345", "LH678", "AF901", "JL234", "SQ567", "EK890"]
    times_dep = ["9:00 AM", "11:30 AM", "2:15 PM", "4:45 PM", "7:00 PM", "6:30 AM", "8:15 AM", "1:00 PM", "3:30 PM", "5:00 PM"]
    terminals = ["A", "B", "C", "D", "E", "1", "2", "3", "4", "5"]
    meetings = ["team", "project", "client", "board", "standup", "retrospective", "planning", "review", "sync", "all-hands"]
    times_meet = ["2:00", "3:00", "4:00", "10:00", "11:00", "1:00", "9:00", "12:00", "3:30", "4:30"]

    for i in range(50):
        template, src, tags = noise_topics[i % len(noise_topics)]
        if i % 5 == 0:
            content = template.format(weather[i % 10], days[i % 10])
        elif i % 5 == 1:
            content = template.format(tasks[i % 10], days[i % 10])
        elif i % 5 == 2:
            content = template.format(restaurants[i % 10], streets[i % 10])
        elif i % 5 == 3:
            content = template.format(flights[i % 10], times_dep[i % 10], terminals[i % 10])
        else:
            content = template.format(meetings[i % 10], times_meet[i % 10])
        noise_memories.append((content, src, tags))

    # Semi-relevant memories (indices 75-99)
    semi_memories = [
        ("Python's asyncio library enables concurrent I/O operations.", "system", ["coding"]),
        ("The team decided to use pytest for all automated testing.", "conversation", ["implementation"]),
        ("Docker containers provide reproducible deployment environments.", "system", ["deployment"]),
        ("The GPU cluster has 8x A100 GPUs available for training.", "system", ["infrastructure"]),
        ("Git branching strategy should follow trunk-based development.", "conversation", ["workflow"]),
        ("The CI/CD pipeline runs on every pull request automatically.", "system", ["workflow"]),
        ("Code review comments should be addressed within 24 hours.", "conversation", ["workflow"]),
        ("The documentation should follow Google-style docstrings.", "conversation", ["coding"]),
        ("Unit test coverage should exceed 80% for all modules.", "conversation", ["testing"]),
        ("The database backup runs every 6 hours automatically.", "system", ["infrastructure"]),
        ("API rate limiting is set to 100 requests per minute.", "system", ["infrastructure"]),
        ("The logging format uses structured JSON for easier parsing.", "system", ["implementation"]),
        ("Performance benchmarks should be run on consistent hardware.", "conversation", ["evaluation"]),
        ("The error handling strategy uses custom exception classes.", "system", ["coding"]),
        ("Memory profiling shows the application uses 2GB peak RAM.", "observation", ["infrastructure"]),
        ("The caching layer reduces API latency by 60%.", "observation", ["infrastructure"]),
        ("Type hints should be used throughout the codebase.", "conversation", ["coding"]),
        ("The monitoring dashboard tracks CPU, memory, and API metrics.", "system", ["infrastructure"]),
        ("Load testing with 1000 concurrent users shows 99.5% uptime.", "observation", ["infrastructure"]),
        ("The feature flag system enables gradual rollouts.", "system", ["deployment"]),
        ("Code formatting uses Black with line length of 100.", "conversation", ["coding"]),
        ("The staging environment mirrors production configuration.", "system", ["deployment"]),
        ("Dependency updates should be reviewed monthly for security.", "conversation", ["workflow"]),
        ("The backup strategy includes both incremental and full backups.", "system", ["infrastructure"]),
        ("Integration tests cover all API endpoints.", "conversation", ["testing"]),
    ]

    all_memories = research_memories + noise_memories + semi_memories

    queries = [
        ("What is FAMM's approach to memory management?", [1, 2, 10, 24]),
        ("How does the forgetting mechanism work?", [5, 6, 8, 16, 22]),
        ("What are the experimental results?", [7, 8, 9, 17, 20]),
        ("Describe the evaluation methodology.", [13, 14, 21]),
        ("How does goal-aware retrieval differ from standard RAG?", [3, 7, 10, 11]),
        ("What implementation technologies are used?", [11, 12, 15, 23]),
        ("Explain the memory lifecycle states.", [24, 18]),
        ("What baselines are compared against FAMM?", [2, 5]),
        ("How is the utility predictor trained?", [9, 19, 20]),
        ("What is the consolidation strategy?", [17, 6]),
    ]

    return all_memories, queries


def run_scaled_experiment():
    """Run experiment with 100 memories."""
    logger.info("Running scaled experiment (100 memories)...")

    es = get_embedding_service()
    all_memories, queries = create_scaled_scenario()
    
    goals = [
        "Complete the research paper on memory management for LLM agents",
        "Run experiments comparing FAMM against baseline systems",
        "Write the methodology and evaluation sections",
    ]

    temp_dirs = []
    def mkd(name):
        d = tempfile.mkdtemp(prefix=f"famm_{name}_")
        temp_dirs.append(d)
        return d

    # ── FAMM ──
    config = FAMMConfig.default()
    config.vector_db.chroma.persist_directory = mkd("famm")
    vs = ChromaAdapter(config.vector_db.chroma)
    eb = EventBus()
    fup = FutureUtilityPredictor(config.future_utility_predictor, es)
    mm = MemoryManager(config, vs, es, eb, fup)
    retriever = GoalAwareRetriever(config.goal_retrieval, vs, es, eb)

    stored_ids = []
    for content, src_str, tags in all_memories:
        src_map = {"reflection": SourceType.REFLECTION, "observation": SourceType.OBSERVATION,
                   "conversation": SourceType.CONVERSATION, "system": SourceType.SYSTEM}
        r = mm.store(content, source_type=src_map.get(src_str, SourceType.CONVERSATION),
                     goal_tags=tags, goal_context=goals)
        stored_ids.append(r.id)

    for _ in range(50):
        mm.step()

    famm_results = {}
    for query, rel_indices in queries:
        rel_ids = {stored_ids[i] for i in rel_indices if i < len(stored_ids)}
        results = retriever.retrieve(query, goals=goals, top_k=10)
        ret_ids = [r.memory.id for r in results]
        p = precision_at_k(ret_ids, rel_ids, 10)
        r = recall_at_k(ret_ids, rel_ids, 10)
        famm_results[query[:40]] = {"p@10": p, "r@10": r, "f1": f1_score(p, r), "ndcg": ndcg_at_k(ret_ids, rel_ids, 10)}

    # ── Baselines ──
    baseline_systems = {
        "SimilarityOnly": SimilarityOnlyBaseline(ChromaAdapter(ChromaConfig(persist_directory=mkd("sim"), collection_name="sim")), es),
        "ImportanceScoring": ImportanceScoringBaseline(ChromaAdapter(ChromaConfig(persist_directory=mkd("imp"), collection_name="imp")), es),
        "EbbinghausDecay": EbbinghausDecayBaseline(ChromaAdapter(ChromaConfig(persist_directory=mkd("ebb"), collection_name="ebb")), es),
    }

    baseline_results = {}
    for name, sys in baseline_systems.items():
        bl_ids = []
        for content, src_str, tags in all_memories:
            r = sys.store(content, source_type=src_str)
            bl_ids.append(r.id)
        for _ in range(50):
            sys.step()

        bl_metrics = {}
        for query, rel_indices in queries:
            rel_ids = {bl_ids[i] for i in rel_indices if i < len(bl_ids)}
            results = sys.retrieve(query, top_k=10)
            ret_ids = [r.id for r in results]
            p = precision_at_k(ret_ids, rel_ids, 10)
            r_val = recall_at_k(ret_ids, rel_ids, 10)
            bl_metrics[query[:40]] = {"p@10": p, "r@10": r_val, "f1": f1_score(p, r_val), "ndcg": ndcg_at_k(ret_ids, rel_ids, 10)}
        baseline_results[name] = bl_metrics

    # Cleanup
    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    return famm_results, baseline_results


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Comparative Bar Chart
# ═══════════════════════════════════════════════════════════

def generate_comparison_chart(famm_results, baseline_results):
    """Generate paper Figure 3: System comparison bar chart."""
    logger.info("Generating Figure 3: Comparison chart...")

    systems = {"FAMM": famm_results}
    systems.update(baseline_results)

    # Aggregate per system
    agg = {}
    for name, qr in systems.items():
        metrics = list(qr.values())
        agg[name] = {
            "P@10": np.mean([m["p@10"] for m in metrics]),
            "R@10": np.mean([m["r@10"] for m in metrics]),
            "F1@10": np.mean([m["f1"] for m in metrics]),
            "NDCG@10": np.mean([m["ndcg"] for m in metrics]),
        }

    metric_names = ["P@10", "R@10", "F1@10", "NDCG@10"]
    system_names = list(agg.keys())
    colors = ['#2ecc71', '#3498db', '#e67e22', '#e74c3c']

    x = np.arange(len(metric_names))
    width = 0.18
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    for i, (sys_name, color) in enumerate(zip(system_names, colors)):
        vals = [agg[sys_name][m] for m in metric_names]
        bars = ax.bar(x + i * width, vals, width, label=sys_name, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Metric', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('FAMM vs Baselines — Scaled Experiment (100 Memories, 10 Queries)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.legend(fontsize=11, loc='upper left')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_comparison_scaled.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig3_comparison_scaled.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("  → Saved fig3_comparison_scaled.png/pdf")

    return agg


# ═══════════════════════════════════════════════════════════
# ABLATION STUDY
# ═══════════════════════════════════════════════════════════

def run_ablation_study():
    """Run ablation: full FAMM vs removing each component."""
    logger.info("Running ablation study...")

    es = get_embedding_service()
    all_memories, queries = create_scaled_scenario()
    goals = ["Complete the research paper", "Run experiments", "Write methodology"]

    configs = {
        "FAMM (Full)": {"use_predictor": True, "use_goals": True, "use_decay": True},
        "– No Utility Predictor": {"use_predictor": False, "use_goals": True, "use_decay": True},
        "– No Goal Retrieval": {"use_predictor": True, "use_goals": False, "use_decay": True},
        "– No Adaptive Decay": {"use_predictor": True, "use_goals": True, "use_decay": False},
        "– None (Similarity Only)": {"use_predictor": False, "use_goals": False, "use_decay": False},
    }

    ablation_results = {}
    temp_dirs = []

    for config_name, flags in configs.items():
        d = tempfile.mkdtemp(prefix="famm_abl_")
        temp_dirs.append(d)

        cfg = FAMMConfig.default()
        cfg.vector_db.chroma.persist_directory = d
        vs = ChromaAdapter(cfg.vector_db.chroma)
        eb = EventBus()

        fup = FutureUtilityPredictor(cfg.future_utility_predictor, es) if flags["use_predictor"] else None
        mm = MemoryManager(cfg, vs, es, eb, fup)

        stored_ids = []
        for content, src_str, tags in all_memories:
            src_map = {"reflection": SourceType.REFLECTION, "observation": SourceType.OBSERVATION,
                       "conversation": SourceType.CONVERSATION, "system": SourceType.SYSTEM}
            r = mm.store(content, source_type=src_map.get(src_str, SourceType.CONVERSATION),
                         goal_tags=tags, goal_context=goals if flags["use_predictor"] else None)
            stored_ids.append(r.id)

        if flags["use_decay"]:
            for _ in range(50):
                mm.step()

        metrics_list = []
        for query, rel_indices in queries:
            rel_ids = {stored_ids[i] for i in rel_indices if i < len(stored_ids)}

            if flags["use_goals"]:
                retriever = GoalAwareRetriever(cfg.goal_retrieval, vs, es, eb)
                results = retriever.retrieve(query, goals=goals, top_k=10)
                ret_ids = [r.memory.id for r in results]
            else:
                results = mm.retrieve(query, top_k=10)
                ret_ids = [r.id for r in results]

            p = precision_at_k(ret_ids, rel_ids, 10)
            r_val = recall_at_k(ret_ids, rel_ids, 10)
            metrics_list.append({"p": p, "r": r_val, "f1": f1_score(p, r_val), "ndcg": ndcg_at_k(ret_ids, rel_ids, 10)})

        ablation_results[config_name] = {
            "P@10": np.mean([m["p"] for m in metrics_list]),
            "R@10": np.mean([m["r"] for m in metrics_list]),
            "F1@10": np.mean([m["f1"] for m in metrics_list]),
            "NDCG@10": np.mean([m["ndcg"] for m in metrics_list]),
        }

    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    return ablation_results


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Ablation Chart
# ═══════════════════════════════════════════════════════════

def generate_ablation_chart(ablation_results):
    """Generate paper Figure 4: Ablation study results."""
    logger.info("Generating Figure 4: Ablation chart...")

    config_names = list(ablation_results.keys())
    metric_names = ["P@10", "R@10", "F1@10", "NDCG@10"]
    colors_map = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)

    for ax, metric, color in zip(axes, metric_names, colors_map):
        vals = [ablation_results[c][metric] for c in config_names]
        bars = ax.barh(range(len(config_names)), vals, color=color, alpha=0.8, edgecolor='white')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

        ax.set_xlim(0, 1.1)
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_yticks(range(len(config_names)))
        if ax == axes[0]:
            ax.set_yticklabels(config_names, fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()

    fig.suptitle('Ablation Study — Contribution of Each FAMM Module', fontsize=15, fontweight='bold', y=1.02)
    sns.despine()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_ablation.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig4_ablation.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("  → Saved fig4_ablation.png/pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Feature Importance
# ═══════════════════════════════════════════════════════════

def generate_feature_importance():
    """Generate paper Figure 5: Feature contribution analysis."""
    logger.info("Generating Figure 5: Feature importance...")

    feature_names = ["Goal\nSimilarity", "Recency", "Access\nFrequency", "Source\nType", "Entity\nOverlap"]
    weights = [0.35, 0.20, 0.15, 0.15, 0.15]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = ['#2ecc71', '#3498db', '#e67e22', '#9b59b6', '#e74c3c']
    
    bars = ax.bar(feature_names, weights, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{w:.0%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Feature Weight', fontsize=13)
    ax.set_title('Heuristic Scorer Feature Weights', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 0.45)
    ax.grid(True, alpha=0.3, axis='y')
    sns.despine()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_feature_importance.png", dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / "fig5_feature_importance.pdf", bbox_inches='tight')
    plt.close(fig)
    logger.info("  → Saved fig5_feature_importance.png/pdf")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("FAMM — Full Experiment Suite")
    logger.info("=" * 60)

    t0 = time.time()

    # Figure 1: Decay curves (no model needed)
    generate_decay_curves()

    # Figure 2: Utility distribution
    generate_utility_distribution()

    # Scaled experiment
    famm_res, baseline_res = run_scaled_experiment()

    # Figure 3: Comparison chart
    agg = generate_comparison_chart(famm_res, baseline_res)

    # Print table
    logger.info("\n" + "=" * 60)
    logger.info("SCALED EXPERIMENT RESULTS (100 memories, 10 queries)")
    logger.info("=" * 60)
    header = f"{'System':<22} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'NDCG@10':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for sys_name, metrics in agg.items():
        logger.info(f"{sys_name:<22} {metrics['P@10']:>8.4f} {metrics['R@10']:>8.4f} {metrics['F1@10']:>8.4f} {metrics['NDCG@10']:>8.4f}")

    # Ablation study
    ablation = run_ablation_study()

    # Figure 4: Ablation
    generate_ablation_chart(ablation)

    # Print ablation table
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 60)
    header = f"{'Configuration':<30} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'NDCG@10':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for config_name, metrics in ablation.items():
        logger.info(f"{config_name:<30} {metrics['P@10']:>8.4f} {metrics['R@10']:>8.4f} {metrics['F1@10']:>8.4f} {metrics['NDCG@10']:>8.4f}")

    # Figure 5: Feature importance
    generate_feature_importance()

    # Save all results
    all_results = {
        "scaled_experiment": {sys: {q: {k: round(v, 4) for k, v in m.items()} for q, m in qr.items()} for sys, qr in {**{"FAMM": famm_res}, **baseline_res}.items()},
        "scaled_aggregated": {sys: {k: round(v, 4) for k, v in m.items()} for sys, m in agg.items()},
        "ablation": {cfg: {k: round(v, 4) for k, v in m.items()} for cfg, m in ablation.items()},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_path = RESULTS_DIR / "full_experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"\nAll experiments complete in {elapsed:.1f}s")
    logger.info(f"Results: {results_path}")
    logger.info(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
