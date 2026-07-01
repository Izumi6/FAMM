"""
VectorStoreAdapter — Abstract interface for vector database operations.

All vector database backends (ChromaDB, FAISS) implement this interface,
ensuring that the rest of FAMM is agnostic to the storage backend.

This is the Strategy pattern: the Memory Engine depends on this abstraction,
and concrete adapters are injected at runtime based on configuration.

Design Decisions:
- We separate embedding storage from metadata storage. The vector DB stores
  embeddings + document IDs; metadata is stored alongside in the DB's native
  metadata support (or in SQLite for backends that lack it).
- All methods are synchronous. For production-scale async support, this
  interface would be extended with async variants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreAdapter(ABC):
    """
    Abstract base class for vector database adapters.

    Each adapter must implement CRUD operations for vector storage:
    - add: Store embeddings with metadata
    - query: Similarity search
    - get: Retrieve by ID
    - update: Update metadata for existing records
    - delete: Remove records by ID
    - count: Return the number of stored records
    """

    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add records to the vector store.

        Args:
            ids: Unique identifiers for each record.
            embeddings: Dense vector representations.
            documents: Raw text content (stored for retrieval).
            metadatas: Metadata dictionaries for each record.

        Raises:
            ValueError: If input lists have mismatched lengths.
        """
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the vector store for similar records.

        Args:
            query_embedding: Dense vector to search for.
            top_k: Maximum number of results to return.
            where: Optional metadata filter (backend-specific syntax).

        Returns:
            List of result dictionaries, each containing:
                - id: Record identifier
                - document: Raw text content
                - metadata: Stored metadata
                - distance: Similarity distance (lower = more similar)
                - embedding: The stored embedding vector
        """
        ...

    @abstractmethod
    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve records by their IDs.

        Args:
            ids: List of record identifiers to retrieve.

        Returns:
            List of record dictionaries (same format as query results).
            Missing IDs are silently skipped.
        """
        ...

    @abstractmethod
    def update(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Update existing records in the vector store.

        Only provided fields are updated; others remain unchanged.

        Args:
            ids: Identifiers of records to update.
            metadatas: Updated metadata (if any).
            embeddings: Updated embeddings (if any).
            documents: Updated documents (if any).
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """
        Delete records from the vector store.

        Args:
            ids: Identifiers of records to delete.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of records in the store.

        Returns:
            Integer count of stored records.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all records from the store."""
        ...
