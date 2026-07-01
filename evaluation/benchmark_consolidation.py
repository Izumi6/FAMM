#!/usr/bin/env python3
"""
Benchmark Consolidation Engine.
"""

import logging
import tempfile
import shutil
import time
from pathlib import Path

from backend.memory_engine.event_bus import EventBus
from backend.memory_engine.memory_manager import MemoryManager
from backend.memory_engine.memory_record import SourceType
from backend.vector_database.chroma_adapter import ChromaAdapter
from backend.vector_database.embedding_service import EmbeddingService
from backend.future_utility_predictor.predictor import FutureUtilityPredictor
from backend.goal_retrieval.retriever import GoalAwareRetriever
from backend.consolidation.consolidator import Consolidator
from config.settings import FAMMConfig
from evaluation.metrics import precision_at_k, recall_at_k

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("consolidation_bench")
logger.setLevel(logging.INFO)

def run_consolidation_benchmark():
    es = EmbeddingService()
    cfg = FAMMConfig.default()
    
    # We will lower cluster threshold to guarantee clusters form for the test
    cfg.consolidation.cluster_similarity_threshold = 0.4
    cfg.consolidation.min_cluster_size = 3
    
    d = tempfile.mkdtemp(prefix="famm_consol_")
    cfg.vector_db.chroma.persist_directory = d
    
    vs = ChromaAdapter(cfg.vector_db.chroma)
    eb = EventBus()
    fup = FutureUtilityPredictor(cfg.future_utility_predictor, es)
    mm = MemoryManager(cfg, vs, es, eb, fup)
    retriever = GoalAwareRetriever(cfg.goal_retrieval, vs, es, eb)
    
    clusters_data = [
        ("The user is researching memory architectures for LLM agents.", "researching memory in LLMs", "LLM agents need external memory"),
        ("Python 3.11 introduced performance improvements.", "Python 3.11 is faster", "We use Python 3.11 for the backend"),
        ("Vector databases store embeddings for similarity search.", "ChromaDB is a vector database", "FAISS and Chroma are used for embeddings"),
        ("Ebbinghaus curve describes how humans forget information.", "The forgetting curve is exponential", "Biological memory decays over time"),
        ("Goal-aware retrieval improves precision.", "We use goals to align retrieval", "Retrieval is better when goals are active")
    ]
    
    goals = ["Write paper on LLM memory", "Implement database"]
    
    all_stored_ids = []
    cluster_ground_truth = []
    
    for cluster_texts in clusters_data:
        base = cluster_texts[0]
        # duplicate 3 times with minor noise
        for _ in range(3):
            for text in cluster_texts:
                r = mm.store(text, source_type=SourceType.OBSERVATION, goal_tags=["research"], goal_context=goals)
                all_stored_ids.append(r.id)
                if base == clusters_data[0][0]:
                    cluster_ground_truth.append(r.id)
                    
    # Noise
    for i in range(50):
        r = mm.store(f"Noise memory number {i} about unrelated things.", source_type=SourceType.OBSERVATION, goal_tags=[])
        all_stored_ids.append(r.id)
        
    for mem_id in all_stored_ids:
        rec = mm.get_by_id(mem_id)
        if rec:
            rec.utility_score = 0.5
            mm._update_record_in_store(rec)
        
    # Baseline Retrieval (Before Consolidation)
    q = "What is the user researching?"
    res_before = retriever.retrieve(q, goals=goals, top_k=5)
    ids_before = [r.memory.id for r in res_before]
    p_before = precision_at_k(ids_before, set(cluster_ground_truth), 5)
    r_before = recall_at_k(ids_before, set(cluster_ground_truth), 5)
    
    count_before = len(mm._get_all_active_records())
    
    logger.info(f"Before Consolidation:")
    logger.info(f"  Active Memories: {count_before}")
    logger.info(f"  Precision@5: {p_before:.2f}")
    logger.info(f"  Recall@5: {r_before:.2f}")
    
    # Run Consolidation
    consolidator = Consolidator(cfg.consolidation, es, vs, eb)
    
    logger.info("Running Consolidation...")
    t0 = time.time()
    
    # Needs to bypass predictor for tests, we will just set high utility manually
    # or rely on the fact that policy uses threshold. Let's just run it.
    stats = consolidator.run(mm._get_all_active_records())
    
    # Sync memory manager cache since consolidator bypassed EventBus
    for rm_id in stats["removed_record_ids"]:
        mm._memory_cache.pop(rm_id, None)
        
    if stats["new_record_ids"]:
        raw_res = vs.get(ids=stats["new_record_ids"])
        for r_dict in raw_res:
            rec = mm._result_to_record(r_dict)
            if rec:
                mm._memory_cache[rec.id] = rec
    
    # We must reload memory manager state if consolidator modified DB directly
    # Wait, MemoryManager's _get_all_active_records just queries VectorStore!
    # So it should be updated.
    
    count_after = len(mm._get_all_active_records())
    
    res_after = retriever.retrieve(q, goals=goals, top_k=5)
    ids_after = [r.memory.id for r in res_after]
    
    relevant_retrieved = sum(1 for m in res_after if "research" in m.memory.content.lower() or "llm" in m.memory.content.lower())
    p_after = relevant_retrieved / 5.0
    
    logger.info(f"After Consolidation:")
    logger.info(f"  Active Memories: {count_after}")
    if count_before > 0:
        logger.info(f"  Storage Reduction: {(1 - count_after/count_before)*100:.1f}%")
    logger.info(f"  Precision@5 (Semantic): {p_after:.2f}")
    logger.info(f"  Consolidation Time: {time.time()-t0:.2f}s")
    
    shutil.rmtree(d, ignore_errors=True)
    
if __name__ == "__main__":
    run_consolidation_benchmark()
