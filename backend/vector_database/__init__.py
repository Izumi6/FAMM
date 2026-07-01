"""
FAMM Vector Database Package.

Provides abstract storage interface and concrete adapters
for ChromaDB and FAISS, plus the embedding service.
"""

from backend.vector_database.base_adapter import VectorStoreAdapter
from backend.vector_database.embedding_service import EmbeddingService

__all__ = ["VectorStoreAdapter", "EmbeddingService"]
