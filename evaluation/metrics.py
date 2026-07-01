"""
Evaluation Metrics — Standardized metrics for FAMM experiments.

Implements all metrics reported in the paper:
1. Retrieval Precision@K: Fraction of top-K results that are relevant
2. Retrieval Recall@K: Fraction of relevant memories in top-K
3. Memory Efficiency: Active memories / total memories stored
4. Utility Retention: Average utility of surviving memories over time
5. F1-Score: Harmonic mean of precision and recall
6. NDCG@K: Normalized discounted cumulative gain

All metrics return values in [0.0, 1.0] for consistent comparison.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> float:
    """
    Precision@K: fraction of retrieved items that are relevant.

    Args:
        retrieved_ids: Ordered list of retrieved memory IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Number of results to consider (default: all).

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids:
        return 0.0

    relevant_count = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return relevant_count / len(retrieved_ids)


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> float:
    """
    Recall@K: fraction of relevant items that were retrieved.

    Args:
        retrieved_ids: Ordered list of retrieved memory IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Number of results to consider (default: all).

    Returns:
        Recall score in [0.0, 1.0].
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not relevant_ids:
        return 0.0

    retrieved_relevant = sum(1 for rid in retrieved_ids if rid in relevant_ids)
    return retrieved_relevant / len(relevant_ids)


def f1_score(precision: float, recall: float) -> float:
    """
    F1 score: harmonic mean of precision and recall.

    Args:
        precision: Precision value.
        recall: Recall value.

    Returns:
        F1 score in [0.0, 1.0].
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int | None = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain (NDCG@K).

    Measures ranking quality: relevant items should appear earlier.

    Args:
        retrieved_ids: Ordered list of retrieved memory IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: Number of results to consider.

    Returns:
        NDCG score in [0.0, 1.0].
    """
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
    if not retrieved_ids or not relevant_ids:
        return 0.0

    # DCG: sum of 1/log2(rank+1) for each relevant item
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG: all relevant items at top positions
    ideal_k = min(len(relevant_ids), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def memory_efficiency(
    active_count: int,
    total_stored: int,
) -> float:
    """
    Memory efficiency: fraction of stored memories that are active.

    Higher is better — means the system isn't wasting storage on
    useless memories.

    Args:
        active_count: Number of ACTIVE state memories.
        total_stored: Total memories ever stored.

    Returns:
        Efficiency ratio in [0.0, 1.0].
    """
    if total_stored == 0:
        return 1.0
    return active_count / total_stored


def utility_retention(utilities: list[float]) -> float:
    """
    Average utility of surviving memories.

    Higher means the system is preserving high-value memories
    and letting low-value ones decay.

    Args:
        utilities: List of utility scores of active memories.

    Returns:
        Mean utility in [0.0, 1.0].
    """
    if not utilities:
        return 0.0
    return float(np.mean(utilities))


def compute_all_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    active_count: int,
    total_stored: int,
    utilities: list[float],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute all metrics in one call.

    Returns:
        Dict of all metric values.
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)

    return {
        f"precision@{k}": round(p, 4),
        f"recall@{k}": round(r, 4),
        f"f1@{k}": round(f1_score(p, r), 4),
        f"ndcg@{k}": round(ndcg_at_k(retrieved_ids, relevant_ids, k), 4),
        "memory_efficiency": round(memory_efficiency(active_count, total_stored), 4),
        "utility_retention": round(utility_retention(utilities), 4),
    }
