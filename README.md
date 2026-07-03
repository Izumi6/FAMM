# FAMM — Future-Aware Adaptive Memory Management Framework

> **For Long-Term Autonomous LLM Agents**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-82%20passing-brightgreen.svg)]()
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21168000.svg)](https://doi.org/10.5281/zenodo.21168000)
[![License](https://img.shields.io/badge/License-Research-orange.svg)]()

## Abstract

FAMM introduces a **future-aware** approach to memory management for long-term autonomous LLM agents. Unlike existing systems (MemGPT, Mem0, MemoryBank) that rely on semantic similarity, importance heuristics, or uniform temporal decay, FAMM predicts the **future utility** of each memory at write-time and uses this prediction to drive adaptive forgetting, goal-aware retrieval, and intelligent consolidation.

### Key Innovations

| Innovation | What It Does | How It's Different |
|:---|:---|:---|
| **Future Utility Predictor** | Scores memories at write-time for predicted future relevance | Proactive (at creation) vs. reactive (at retrieval) |
| **Goal-Aware Retriever** | Ranks memories by 4 signals: similarity + utility + goal alignment + recency | Context-sensitive vs. query-only |
| **Utility-Conditioned Decay** | Decay rate inversely proportional to utility (80× speed difference) | Content-aware vs. uniform temporal |
| **Adaptive Consolidation** | Clusters and merges related memories by semantic + goal similarity | Reduces storage while preserving quality |

## Architecture

```
FAMM Framework
├── Memory Engine         # Core: MemoryRecord, EventBus, LifecycleController, MemoryManager
├── Future Utility        # Feature extractor, heuristic scorer, learned MLP, predictor
│   Predictor             
├── Goal-Aware            # Goal encoder, multi-signal ranker, retriever
│   Retriever             
├── Forgetting Engine     # Utility decay, Ebbinghaus baseline, scheduler, pruner
├── Consolidation         # Cluster engine, summarizer, policy, consolidator
├── Vector Database       # ChromaDB adapter, FAISS adapter, embedding service
├── Baselines             # NaiveFIFO, SimilarityOnly, ImportanceScoring, EbbinghausDecay
└── Evaluation            # Metrics (P@K, R@K, F1, NDCG), experiment runner
```

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd "FAMM IEEE"
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Run tests (82 tests, ~50s)
pytest tests/ -v

# Run experiments
python run_full_experiments.py

# Results will be in:
# - experiments/results/   (JSON metrics)
# - experiments/figures/   (Paper figures)
```

## Configuration

All parameters are configurable via `config/default.yaml`:

```yaml
future_utility_predictor:
  mode: heuristic  # or "learned" (after training)
  heuristic:
    weight_goal_similarity: 0.35
    weight_recency: 0.20
    weight_access_frequency: 0.15
    weight_source_type: 0.15
    weight_entity_overlap: 0.15

forgetting_engine:
  base_decay_rate: 0.05
  utility_exponent: 2.0     # Higher = more protection for high-utility
  prune_threshold: 0.05
  decay_interval_steps: 10
```

## Core Decay Formula

FAMM's utility-conditioned decay:

```
effective_decay = base_rate × (1 - utility_score)^exponent
```

With defaults (base=0.05, exp=2.0):
- **Utility 0.9**: decay = 0.0005/step (very slow — protected)
- **Utility 0.5**: decay = 0.0125/step (moderate)
- **Utility 0.1**: decay = 0.0405/step (fast — will be forgotten)

This creates an **80× speed difference** between high and low utility memories.

## Project Structure

```
FAMM IEEE/
├── backend/
│   ├── memory_engine/          # Core data structures and orchestrator
│   ├── future_utility_predictor/  # ★ Novel: proactive utility prediction
│   ├── goal_retrieval/         # ★ Novel: goal-aware multi-signal retrieval
│   ├── forgetting_engine/      # Adaptive decay + Ebbinghaus baseline
│   ├── consolidation/          # Semantic clustering and summarization
│   └── vector_database/        # ChromaDB, FAISS, embeddings
├── baselines/                  # 4 comparison systems
├── config/                     # Pydantic configuration models + YAML
├── evaluation/                 # Metrics and experiment runner
├── experiments/                # Results, figures, logs
├── tests/                      # 82 unit + integration tests
│   ├── unit/
│   └── integration/
├── datasets/                   # Benchmark data (LoCoMo, etc.)
├── paper/                      # IEEE manuscript
├── run_experiment.py           # Quick experiment
└── run_full_experiments.py     # Full suite with figures
```

## Tech Stack

| Component | Technology |
|:---|:---|
| Language | Python 3.11+ |
| Configuration | Pydantic v2 + YAML |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2, 384d) |
| Vector DB | ChromaDB (primary) + FAISS (scale testing) |
| ML Model | scikit-learn MLPRegressor (16→8 hidden units) |
| Testing | pytest + pytest-cov |
| Visualization | matplotlib + seaborn |
| LLM (optional) | Ollama (for LLM-based consolidation) |

## Citation

```bibtex
@article{famm2026,
  author={Vakhariya, Suyash and Ipper, Asmita},
  title={FAMM: Future-Aware Adaptive Memory Management Framework 
         for Long-Term Autonomous LLM Agents},
  year={2026},
  doi={10.5281/zenodo.21168000},
  url={https://doi.org/10.5281/zenodo.21168000},
  publisher={Zenodo}
}
```

## License

This project is part of an active research submission. All rights reserved until publication.
