"""
Synthetic Data Generator — For large-scale memory experiments.
"""

import random
from typing import List, Tuple, Dict

def generate_memory_stream(size: int, random_seed: int = 42) -> Tuple[List[Tuple[str, str, List[str]]], List[Tuple[str, List[int]]]]:
    """
    Generates a large-scale procedural memory stream.
    
    Args:
        size: Number of memories to generate (e.g., 1000, 10000).
        random_seed: Seed for reproducibility.
        
    Returns:
        all_memories: List of (content, source_type, tags)
        queries: List of (query_text, relevant_indices)
    """
    random.seed(random_seed)
    
    # 1. Base Signal (Targeted Knowledge)
    # We create a core set of memories that we will query.
    signal_memories = [
        ("The FAMM future utility predictor utilizes both heuristic and MLP models.", "reflection", ["research", "methodology"]),
        ("Standard RAG relies on semantic similarity alone, which degrades at high scale.", "observation", ["research"]),
        ("Memory consolidation uses a Union-Find clustering algorithm.", "system", ["implementation"]),
        ("Utility decay follows an adaptive exponential curve based on utility.", "reflection", ["research"]),
        ("The memory event bus decouples the retriever from the predictor.", "system", ["architecture"]),
        ("The goal-aware retriever combines semantic, utility, alignment, and recency.", "reflection", ["research", "methodology"]),
        ("Python 3.11 is required for Pydantic v2 performance improvements.", "system", ["implementation"]),
        ("Ablation studies show that pure RAG achieves high NDCG on very small datasets.", "observation", ["evaluation"]),
        ("Precision@10 drops by 40% when moving from 100 to 10000 memories in standard RAG.", "observation", ["evaluation"]),
        ("Retrospective labeling is used to train the Future Utility Predictor.", "reflection", ["methodology"])
    ]
    
    all_memories = []
    
    # 2. Add signal to known positions
    # We'll spread the 10 signal memories uniformly across the stream
    signal_positions = [int(i * (size / 10)) for i in range(10)]
    # Ensure they are distinct and valid
    signal_positions = [min(p, size - 1) for p in signal_positions]
    
    current_signal_idx = 0
    
    # 3. Procedural Noise Generation
    subjects = ["The user", "The system", "Agent X", "The process", "Database", "API endpoint", "Container", "Pipeline"]
    actions = ["requested", "completed", "failed to process", "optimized", "scheduled", "deleted", "verified", "cached"]
    objects = ["a new task", "the backup protocol", "user credentials", "the dataset", "meeting notes", "an update", "the configuration", "payment details"]
    contexts = ["yesterday", "in the background", "with high latency", "successfully", "due to timeout", "as expected", "requiring manual review", "silently"]
    sources = ["conversation", "observation", "system", "reflection"]
    tags_pool = ["personal", "work", "admin", "debug", "routine", "misc"]
    
    for i in range(size):
        if current_signal_idx < len(signal_positions) and i == signal_positions[current_signal_idx]:
            all_memories.append(signal_memories[current_signal_idx])
            current_signal_idx += 1
        else:
            # Generate random noise memory
            sub = random.choice(subjects)
            act = random.choice(actions)
            obj = random.choice(objects)
            ctx = random.choice(contexts)
            
            content = f"{sub} {act} {obj} {ctx}. ID-{i}"
            src = random.choice(sources)
            tags = random.sample(tags_pool, k=random.randint(0, 2))
            all_memories.append((content, src, tags))
            
    # 4. Define Queries that explicitly target the signal memories
    queries = [
        ("What models does the future utility predictor use?", [signal_positions[0]]),
        ("Why does standard RAG degrade at scale?", [signal_positions[1]]),
        ("What algorithm is used for memory consolidation?", [signal_positions[2]]),
        ("How does utility decay work?", [signal_positions[3]]),
        ("How are the components decoupled?", [signal_positions[4]]),
        ("What signals does the goal-aware retriever use?", [signal_positions[5]]),
        ("What Python version is required?", [signal_positions[6]]),
        ("When does pure RAG perform well?", [signal_positions[7]]),
        ("How much does Precision@10 drop at 10000 memories?", [signal_positions[8]]),
        ("How is the Future Utility Predictor trained?", [signal_positions[9]])
    ]
    
    return all_memories, queries
