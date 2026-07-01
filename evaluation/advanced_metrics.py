"""
Advanced Metrics Engine — For large-scale FAMM evaluation.
"""

from typing import List, Set, Dict

def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculates MRR (Mean Reciprocal Rank)."""
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def false_positive_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Fraction of retrieved items that are NOT relevant."""
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k:
        return 0.0
    fp_count = sum(1 for rid in retrieved_k if rid not in relevant_ids)
    return fp_count / len(retrieved_k)

def false_negative_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Fraction of relevant items that were NOT retrieved."""
    retrieved_k = set(retrieved_ids[:k])
    if not relevant_ids:
        return 0.0
    fn_count = sum(1 for rid in relevant_ids if rid not in retrieved_k)
    return fn_count / len(relevant_ids)

def storage_utilization(active_records: int, total_records: int) -> float:
    """Fraction of active records relative to total inserted."""
    if total_records == 0:
        return 0.0
    return active_records / total_records

def estimate_token_cost(total_characters: int, cost_per_1M_tokens: float = 0.50) -> float:
    """
    Estimates storage token cost.
    Assuming roughly 4 characters per token.
    """
    estimated_tokens = total_characters / 4.0
    return (estimated_tokens / 1_000_000) * cost_per_1M_tokens
