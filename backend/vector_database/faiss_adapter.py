"""
FAISS Adapter — High-performance vector search backend for FAMM.

FAISS is used as a secondary backend for:
- Scale benchmarking (millions of vectors)
- Performance comparison against ChromaDB
- Ablation studies requiring precise control over indexing

Since FAISS is a search library (not a database), this adapter also
manages metadata persistence separately using a JSON sidecar file.

Design Decisions:
- We use IndexFlatIP (inner product) with normalized embeddings,
  which is equivalent to cosine similarity but faster.
- Metadata is stored in a Python dict keyed by vector position,
  with a separate ID → position mapping for lookups.
- Persistence is manual: index and metadata are saved/loaded to disk.
- This adapter is less feature-rich than ChromaDB (no metadata filtering
  in queries) but significantly faster for large-scale retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.vector_database.base_adapter import VectorStoreAdapter
from config.settings import FAISSConfig

logger = logging.getLogger(__name__)


class FAISSAdapter(VectorStoreAdapter):
    """
    FAISS implementation of the VectorStoreAdapter interface.

    Provides high-performance vector similarity search. Metadata
    is managed separately in a JSON sidecar file.

    Attributes:
        config: FAISS-specific configuration.
        index: FAISS index instance.
        id_to_pos: Mapping from record ID to index position.
        pos_to_id: Reverse mapping from position to record ID.
        metadata_store: Metadata for each record, keyed by ID.
        document_store: Document content, keyed by ID.
    """

    def __init__(self, config: FAISSConfig | None = None, dimension: int = 384) -> None:
        """
        Initialize FAISS adapter.

        Args:
            config: FAISS configuration. Uses defaults if None.
            dimension: Embedding vector dimensionality. Default 384
                       matches all-MiniLM-L6-v2.
        """
        self.config = config or FAISSConfig()
        self.dimension = dimension

        persist_path = Path(self.config.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self._index_path = persist_path / "index.faiss"
        self._meta_path = persist_path / "metadata.json"

        # ID ↔ position mappings
        self.id_to_pos: dict[str, int] = {}
        self.pos_to_id: dict[int, str] = {}

        # Sidecar stores
        self.metadata_store: dict[str, dict[str, Any]] = {}
        self.document_store: dict[str, str] = {}
        self.embedding_store: dict[str, list[float]] = {}

        # Load existing index or create new
        if self._index_path.exists() and self._meta_path.exists():
            self._load()
        else:
            self.index = self._create_index()

        logger.info(
            "FAISS adapter initialized. Index type: %s, Dimension: %d, Records: %d",
            self.config.index_type,
            self.dimension,
            self.index.ntotal,
        )

    def _create_index(self) -> faiss.Index:
        """Create a new FAISS index based on configuration."""
        if self.config.index_type == "FlatIP":
            return faiss.IndexFlatIP(self.dimension)
        elif self.config.index_type == "FlatL2":
            return faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.config.index_type}")

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Add records to the FAISS index.

        Args:
            ids: Unique identifiers.
            embeddings: Dense vectors.
            documents: Raw text content.
            metadatas: Metadata dicts.

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

        vectors = np.array(embeddings, dtype=np.float32)
        start_pos = self.index.ntotal

        self.index.add(vectors)

        for i, record_id in enumerate(ids):
            pos = start_pos + i
            self.id_to_pos[record_id] = pos
            self.pos_to_id[pos] = record_id
            self.metadata_store[record_id] = metadatas[i]
            self.document_store[record_id] = documents[i]
            self.embedding_store[record_id] = embeddings[i]

        self._save()
        logger.debug("Added %d records to FAISS index", len(ids))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query FAISS for similar vectors.

        Note: FAISS does not support native metadata filtering.
        The `where` parameter triggers post-retrieval filtering,
        which requires fetching more candidates initially.

        Args:
            query_embedding: Dense vector to search for.
            top_k: Maximum results to return.
            where: Optional metadata filter (applied post-retrieval).

        Returns:
            List of result dicts sorted by similarity.
        """
        if self.index.ntotal == 0:
            return []

        # If filtering, fetch extra candidates to compensate for filtering loss
        fetch_k = min(top_k * 3, self.index.ntotal) if where else min(top_k, self.index.ntotal)

        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, fetch_k)

        results = []
        for i in range(len(indices[0])):
            pos = int(indices[0][i])
            if pos < 0:  # FAISS returns -1 for empty slots
                continue

            record_id = self.pos_to_id.get(pos)
            if record_id is None:
                continue

            metadata = self.metadata_store.get(record_id, {})

            # Apply post-retrieval metadata filter
            if where and not self._matches_filter(metadata, where):
                continue

            record: dict[str, Any] = {
                "id": record_id,
                "document": self.document_store.get(record_id, ""),
                "metadata": metadata,
                "distance": float(distances[0][i]),
                "embedding": self.embedding_store.get(record_id, []),
            }
            results.append(record)

            if len(results) >= top_k:
                break

        return results

    def get(self, ids: list[str]) -> list[dict[str, Any]]:
        """
        Retrieve records by ID.

        Args:
            ids: Record identifiers.

        Returns:
            List of record dicts. Missing IDs are skipped.
        """
        results = []
        for record_id in ids:
            if record_id not in self.id_to_pos:
                continue

            record: dict[str, Any] = {
                "id": record_id,
                "document": self.document_store.get(record_id, ""),
                "metadata": self.metadata_store.get(record_id, {}),
                "embedding": self.embedding_store.get(record_id, []),
            }
            results.append(record)

        return results

    def update(
        self,
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: list[list[float]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """
        Update metadata and/or documents for existing records.

        Note: FAISS does not support in-place embedding updates.
        If embeddings are provided, we rebuild the index.

        Args:
            ids: Record identifiers.
            metadatas: Updated metadata (if any).
            embeddings: Updated embeddings (if any) — triggers rebuild.
            documents: Updated documents (if any).
        """
        if not ids:
            return

        for i, record_id in enumerate(ids):
            if record_id not in self.id_to_pos:
                logger.warning("Record %s not found for update, skipping", record_id)
                continue

            if metadatas is not None and i < len(metadatas):
                self.metadata_store[record_id] = metadatas[i]
            if documents is not None and i < len(documents):
                self.document_store[record_id] = documents[i]
            if embeddings is not None and i < len(embeddings):
                self.embedding_store[record_id] = embeddings[i]

        # If embeddings were updated, rebuild the entire index
        if embeddings is not None:
            self._rebuild_index()

        self._save()
        logger.debug("Updated %d records in FAISS", len(ids))

    def delete(self, ids: list[str]) -> None:
        """
        Delete records by ID.

        Since FAISS IndexFlat doesn't support removal, we rebuild
        the index excluding the deleted records.

        Args:
            ids: Record identifiers to delete.
        """
        if not ids:
            return

        ids_set = set(ids)
        for record_id in ids_set:
            self.id_to_pos.pop(record_id, None)
            self.metadata_store.pop(record_id, None)
            self.document_store.pop(record_id, None)
            self.embedding_store.pop(record_id, None)

        # Remove from reverse mapping
        self.pos_to_id = {
            pos: rid for pos, rid in self.pos_to_id.items()
            if rid not in ids_set
        }

        self._rebuild_index()
        self._save()
        logger.debug("Deleted %d records from FAISS", len(ids))

    def count(self) -> int:
        """Return the number of records."""
        return len(self.id_to_pos)

    def clear(self) -> None:
        """Remove all records and reset the index."""
        self.index = self._create_index()
        self.id_to_pos.clear()
        self.pos_to_id.clear()
        self.metadata_store.clear()
        self.document_store.clear()
        self.embedding_store.clear()
        self._save()
        logger.info("FAISS index cleared")

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, str(self._index_path))

        meta = {
            "id_to_pos": self.id_to_pos,
            "pos_to_id": {str(k): v for k, v in self.pos_to_id.items()},
            "metadata_store": self.metadata_store,
            "document_store": self.document_store,
            "embedding_store": self.embedding_store,
            "dimension": self.dimension,
        }

        with open(self._meta_path, "w") as f:
            json.dump(meta, f)

    def _load(self) -> None:
        """Load index and metadata from disk."""
        self.index = faiss.read_index(str(self._index_path))

        with open(self._meta_path, "r") as f:
            meta = json.load(f)

        self.id_to_pos = meta.get("id_to_pos", {})
        self.pos_to_id = {int(k): v for k, v in meta.get("pos_to_id", {}).items()}
        self.metadata_store = meta.get("metadata_store", {})
        self.document_store = meta.get("document_store", {})
        self.embedding_store = meta.get("embedding_store", {})
        self.dimension = meta.get("dimension", self.dimension)

        logger.info("Loaded FAISS index from disk. Records: %d", self.index.ntotal)

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from the embedding store."""
        self.index = self._create_index()
        self.id_to_pos.clear()
        self.pos_to_id.clear()

        if not self.embedding_store:
            return

        all_ids = list(self.embedding_store.keys())
        all_embeddings = [self.embedding_store[rid] for rid in all_ids]

        vectors = np.array(all_embeddings, dtype=np.float32)
        self.index.add(vectors)

        for i, record_id in enumerate(all_ids):
            self.id_to_pos[record_id] = i
            self.pos_to_id[i] = record_id

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
        """
        Check if metadata matches a simple equality filter.

        This is a simplified filter implementation for FAISS
        (which lacks native metadata filtering unlike ChromaDB).

        Args:
            metadata: Record metadata.
            where: Filter conditions (key=value equality).

        Returns:
            True if all filter conditions are satisfied.
        """
        for key, value in where.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
