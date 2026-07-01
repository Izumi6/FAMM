"""
ChromaDB Adapter — Primary vector database backend for FAMM.

ChromaDB is used as the default storage backend because it provides:
- Built-in persistence to disk
- Native metadata filtering (essential for utility-score and state filters)
- Collection management
- Simple API for rapid research iteration

This adapter implements the VectorStoreAdapter interface, keeping the
rest of FAMM decoupled from the storage backend.

Design Decisions:
- We store metadata as flat key-value pairs (ChromaDB requirement).
- Goal tags are stored as comma-separated strings and parsed on retrieval.
- Embeddings are stored alongside documents in the same collection.
- We use cosine similarity as the distance function (consistent with
  normalized Sentence Transformer embeddings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.vector_database.base_adapter import VectorStoreAdapter
from config.settings import ChromaConfig

logger = logging.getLogger(__name__)


class ChromaAdapter(VectorStoreAdapter):
    """
    ChromaDB implementation of the VectorStoreAdapter interface.

    Provides persistent vector storage with metadata filtering
    for the FAMM memory engine.

    Attributes:
        config: ChromaDB-specific configuration.
        client: ChromaDB persistent client instance.
        collection: Active ChromaDB collection.

    Example:
        >>> config = ChromaConfig(persist_directory="./test_chroma")
        >>> adapter = ChromaAdapter(config)
        >>> adapter.add(
        ...     ids=["mem_1"],
        ...     embeddings=[[0.1, 0.2, 0.3]],
        ...     documents=["Test memory"],
        ...     metadatas=[{"utility_score": 0.8}],
        ... )
        >>> results = adapter.query([0.1, 0.2, 0.3], top_k=1)
        >>> results[0]["document"]
        'Test memory'
    """

    def __init__(self, config: ChromaConfig | None = None) -> None:
        """
        Initialize ChromaDB adapter with persistent storage.

        Args:
            config: ChromaDB configuration. Uses defaults if None.
        """
        self.config = config or ChromaConfig()

        # Ensure persistence directory exists
        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB adapter initialized. Collection: '%s', Records: %d",
            self.config.collection_name,
            self.collection.count(),
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add records to ChromaDB.

        Args:
            ids: Unique identifiers for each record.
            embeddings: Dense vector representations.
            documents: Raw text content.
            metadatas: Metadata dictionaries (must contain only scalar values).

        Raises:
            ValueError: If input lists have mismatched lengths.
        """
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError(
                f"Input length mismatch: ids={len(ids)}, embeddings={len(embeddings)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}"
            )

        if not ids:
            return

        # ChromaDB requires metadata values to be str, int, float, or bool
        sanitized_metadatas = [self._sanitize_metadata(m) for m in metadatas]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=sanitized_metadatas,
        )

        logger.debug("Added %d records to ChromaDB", len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query ChromaDB for similar records.

        Args:
            query_embedding: Dense vector to search for.
            top_k: Maximum number of results.
            where: Optional ChromaDB where filter (e.g., {"state": "active"}).

        Returns:
            List of result dicts with id, document, metadata, distance, embedding.
        """
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count()) if self.collection.count() > 0 else top_k,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }

        if where:
            query_params["where"] = where

        if self.collection.count() == 0:
            return []

        results = self.collection.query(**query_params)

        return self._format_results(results)

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve records by ID from ChromaDB.

        Args:
            ids: List of record identifiers.

        Returns:
            List of record dicts. Missing IDs are silently skipped.
        """
        if not ids:
            return []

        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas", "embeddings"],
        )

        output = []
        for i in range(len(results["ids"])):
            record: dict[str, Any] = {
                "id": results["ids"][i],
                "metadata": results["metadatas"][i] if results.get("metadatas") is not None else {},
                "embedding": results["embeddings"][i] if results.get("embeddings") is not None else [],
                "document": results["documents"][i] if results.get("documents") is not None else "",
            }
            output.append(record)

        return output

    def update(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Update existing records in ChromaDB.

        Args:
            ids: Identifiers of records to update.
            metadatas: Updated metadata (if any).
            embeddings: Updated embeddings (if any).
            documents: Updated documents (if any).
        """
        if not ids:
            return

        update_params: dict[str, Any] = {"ids": ids}

        if metadatas is not None:
            update_params["metadatas"] = [self._sanitize_metadata(m) for m in metadatas]
        if embeddings is not None:
            update_params["embeddings"] = embeddings
        if documents is not None:
            update_params["documents"] = documents

        self.collection.update(**update_params)
        logger.debug("Updated %d records in ChromaDB", len(ids))

    def delete(self, ids: list[str]) -> None:
        """
        Delete records from ChromaDB.

        Args:
            ids: Identifiers of records to delete.
        """
        if not ids:
            return

        self.collection.delete(ids=ids)
        logger.debug("Deleted %d records from ChromaDB", len(ids))

    def count(self) -> int:
        """Return the total number of records in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Remove all records by deleting and recreating the collection."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection '%s' cleared", self.config.collection_name)

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Ensure all metadata values are ChromaDB-compatible types.

        ChromaDB only supports str, int, float, and bool as metadata values.
        Lists and None values are converted to strings.
        """
        sanitized: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = ",".join(str(v) for v in value)
            else:
                sanitized[key] = str(value)
        return sanitized

    @staticmethod
    def _format_results(raw_results: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Convert ChromaDB's nested result format to flat record dicts.

        ChromaDB returns results as parallel lists grouped by query.
        We flatten these into a list of individual records.
        """
        output = []
        if not raw_results["ids"] or not raw_results["ids"][0]:
            return output

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0] if raw_results.get("documents") else [None] * len(ids)
        metadatas = raw_results["metadatas"][0] if raw_results.get("metadatas") else [{}] * len(ids)
        distances = raw_results["distances"][0] if raw_results.get("distances") else [0.0] * len(ids)
        embeddings = raw_results["embeddings"][0] if raw_results.get("embeddings") else [None] * len(ids)

        for i in range(len(ids)):
            record: dict[str, Any] = {
                "id": ids[i],
                "document": documents[i],
                "metadata": metadatas[i] or {},
                "distance": distances[i],
                "embedding": embeddings[i],
            }
            output.append(record)

        return output
