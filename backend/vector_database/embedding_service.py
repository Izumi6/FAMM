"""
EmbeddingService — Dense vector encoding for FAMM.

Wraps Sentence Transformers to provide a clean interface for
encoding text into dense vectors. Used by:
- Memory Engine (at write time)
- Goal-Aware Retriever (at query time)
- Goal Encoder (for encoding goal descriptions)

Design Decisions:
- We use a singleton-style lazy loading pattern to avoid loading
  the model multiple times across modules.
- Normalization is enabled by default so that cosine similarity
  can be computed via inner product (faster in FAISS FlatIP).
- Batch encoding is supported for efficiency during bulk operations
  (e.g., dataset preprocessing, consolidation).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from config.settings import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for encoding text into dense vector embeddings.

    Uses Sentence Transformers models for high-quality semantic
    representations. Supports both single and batch encoding.

    Attributes:
        config: Embedding configuration (model name, device, etc.).
        model: Lazy-loaded SentenceTransformer instance.
        dimension: Dimensionality of the embedding vectors.

    Example:
        >>> config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
        >>> service = EmbeddingService(config)
        >>> embedding = service.encode("The agent completed the task.")
        >>> len(embedding)
        384
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """
        Initialize the embedding service.

        Args:
            config: Embedding configuration. Uses defaults if None.
        """
        self.config = config or EmbeddingConfig()
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

    @property
    def model(self) -> "SentenceTransformer":
        """
        Lazy-load the Sentence Transformer model.

        The model is loaded on first access to avoid slow imports
        at module load time. Subsequent accesses return the cached instance.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: %s", self.config.model_name)
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
            )
            self._dimension = self._model.get_embedding_dimension()
            logger.info(
                "Embedding model loaded. Dimension: %d, Device: %s",
                self._dimension,  # type: ignore[arg-type]
                self.config.device,
            )
        return self._model

    @property
    def dimension(self) -> int:
        """
        Return the dimensionality of embedding vectors.

        Forces model loading if not yet loaded.
        """
        if self._dimension is None:
            _ = self.model  # Trigger lazy load
        assert self._dimension is not None
        return self._dimension

    def encode(self, text: str) -> list[float]:
        """
        Encode a single text string into a dense vector.

        Args:
            text: Input text to encode.

        Returns:
            Dense vector as a list of floats.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot encode empty text")

        embedding = self.model.encode(
            text,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Encode multiple texts into dense vectors.

        More efficient than calling encode() in a loop due to
        batched inference on the underlying model.

        Args:
            texts: List of input texts to encode.

        Returns:
            List of dense vectors, one per input text.

        Raises:
            ValueError: If any text is empty or the list is empty.
        """
        if not texts:
            raise ValueError("Cannot encode empty text list")

        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"Text at index {i} is empty")

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.config.normalize,
            batch_size=self.config.batch_size,
            show_progress_bar=len(texts) > 100,
        )

        return embeddings.tolist()

    def compute_similarity(
        self,
        embedding_a: list[float],
        embedding_b: list[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        If embeddings are normalized (default), this is equivalent
        to the dot product.

        Args:
            embedding_a: First embedding vector.
            embedding_b: Second embedding vector.

        Returns:
            Cosine similarity score in [-1.0, 1.0].
        """
        vec_a = np.array(embedding_a)
        vec_b = np.array(embedding_b)

        if self.config.normalize:
            # Normalized vectors: dot product = cosine similarity
            return float(np.dot(vec_a, vec_b))
        else:
            # General case: compute cosine similarity
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def compute_batch_similarity(
        self,
        query_embedding: list[float],
        candidate_embeddings: list[list[float]],
    ) -> list[float]:
        """
        Compute similarity between a query and multiple candidates.

        Args:
            query_embedding: Query vector.
            candidate_embeddings: List of candidate vectors.

        Returns:
            List of similarity scores, one per candidate.
        """
        query = np.array(query_embedding)
        candidates = np.array(candidate_embeddings)

        if self.config.normalize:
            similarities = candidates @ query
        else:
            query_norm = np.linalg.norm(query)
            candidate_norms = np.linalg.norm(candidates, axis=1)
            denominator = query_norm * candidate_norms
            denominator = np.where(denominator == 0, 1.0, denominator)
            similarities = (candidates @ query) / denominator

        return similarities.tolist()
