"""
MemoryRecord — Core data structure for the FAMM framework.

Each memory in FAMM is represented as a MemoryRecord, which encapsulates:
- The raw content and its dense vector embedding
- Temporal metadata (creation time, last access, access count)
- Future utility score (predicted relevance to upcoming tasks)
- Goal tags (which agent goals this memory is associated with)
- Adaptive decay rate (conditioned on utility, not uniform)
- Lifecycle state (active → stale → archived → deleted)
- Consolidation tracking (group ID if merged with related memories)

Design Decisions:
- We use a frozen-style approach with explicit mutation methods rather than
  a fully mutable dataclass, to maintain auditability of state changes.
- The utility_score is set at creation time by the Future Utility Predictor
  and updated during decay cycles and reinforcement (on access).
- decay_rate is per-memory and adaptive, unlike MemoryBank's uniform Ebbinghaus rate.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """
    Classification of how a memory was created.

    Used by the Future Utility Predictor as a feature for scoring.
    Different source types carry different prior probabilities of future relevance.
    """

    CONVERSATION = "conversation"   # From agent-user dialogue
    OBSERVATION = "observation"     # From tool use / environment observation
    REFLECTION = "reflection"       # From agent self-reflection
    CONSOLIDATED = "consolidated"   # From memory consolidation (merged records)
    SYSTEM = "system"               # System-injected knowledge


class MemoryState(str, Enum):
    """
    Lifecycle state of a memory record.

    State transitions:
        ACTIVE → STALE → ARCHIVED → DELETED

    - ACTIVE: Recently created or accessed; eligible for retrieval.
    - STALE: Utility has decayed below stale threshold; lower retrieval priority.
    - ARCHIVED: Moved to cold storage; only retrieved if explicitly requested.
    - DELETED: Marked for permanent removal.
    """

    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryRecord(BaseModel):
    """
    Core data structure representing a single memory unit in FAMM.

    Attributes:
        id: Unique identifier (UUID4).
        content: Raw text content of the memory.
        embedding: Dense vector representation (set by EmbeddingService).
        created_at: UTC timestamp of creation.
        last_accessed_at: UTC timestamp of most recent retrieval.
        access_count: Number of times this memory has been retrieved.
        source_type: How this memory was created.
        utility_score: Predicted future utility in [0.0, 1.0].
        goal_tags: List of goal identifiers this memory is associated with.
        decay_rate: Per-memory adaptive decay rate.
        state: Current lifecycle state.
        consolidation_group: Group ID if this memory has been consolidated.
        metadata: Extensible key-value metadata.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., min_length=1, description="Raw text content of the memory")
    embedding: list[float] = Field(
        default_factory=list,
        description="Dense vector embedding; empty until set by EmbeddingService",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = Field(default=0, ge=0)
    source_type: SourceType = Field(default=SourceType.CONVERSATION)
    utility_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Predicted future utility; set by FutureUtilityPredictor",
    )
    goal_tags: list[str] = Field(
        default_factory=list,
        description="Goal identifiers this memory is associated with",
    )
    decay_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Per-memory adaptive decay rate; inversely proportional to utility",
    )
    state: MemoryState = Field(default=MemoryState.ACTIVE)
    consolidation_group: str | None = Field(
        default=None,
        description="Group ID if consolidated with other memories",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ─────────────────────────────────────────────
    # Mutation Methods (explicit state changes)
    # ─────────────────────────────────────────────

    def record_access(self) -> None:
        """
        Record that this memory was retrieved.

        Updates last_accessed_at and increments access_count.
        Called by the Goal-Aware Retriever when this memory is returned.
        """
        self.last_accessed_at = datetime.now(timezone.utc)
        self.access_count += 1

    def reinforce(self, boost: float = 0.1) -> None:
        """
        Reinforce this memory's utility score after successful retrieval.

        When a memory is retrieved and proves useful (e.g., the agent
        uses it in its response), we increase its utility score to
        slow future decay.

        Args:
            boost: Amount to increase utility_score (clamped to [0, 1]).
        """
        self.utility_score = min(1.0, self.utility_score + boost)
        self.record_access()

    def apply_decay(self, effective_decay: float) -> None:
        """
        Apply decay to this memory's utility score.

        The effective_decay is computed by the Forgetting Engine
        based on the memory's current utility and the configured
        decay formula.

        Args:
            effective_decay: Amount to subtract from utility_score (≥ 0).
        """
        self.utility_score = max(0.0, self.utility_score - effective_decay)

    def transition_to(self, new_state: MemoryState) -> None:
        """
        Transition this memory to a new lifecycle state.

        Validates that the transition is legal according to the
        lifecycle: ACTIVE → STALE → ARCHIVED → DELETED.

        Args:
            new_state: Target state.

        Raises:
            ValueError: If the transition is not permitted.
        """
        valid_transitions: dict[MemoryState, list[MemoryState]] = {
            MemoryState.ACTIVE: [MemoryState.STALE, MemoryState.ARCHIVED, MemoryState.DELETED],
            MemoryState.STALE: [MemoryState.ACTIVE, MemoryState.ARCHIVED, MemoryState.DELETED],
            MemoryState.ARCHIVED: [MemoryState.ACTIVE, MemoryState.DELETED],
            MemoryState.DELETED: [],  # Terminal state
        }

        if new_state not in valid_transitions.get(self.state, []):
            raise ValueError(
                f"Invalid state transition: {self.state.value} → {new_state.value}"
            )
        self.state = new_state

    def age_seconds(self) -> float:
        """Return the age of this memory in seconds since creation."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def seconds_since_access(self) -> float:
        """Return seconds since this memory was last accessed."""
        return (datetime.now(timezone.utc) - self.last_accessed_at).total_seconds()

    def to_storage_dict(self) -> dict[str, Any]:
        """
        Serialize to a dictionary suitable for vector DB metadata storage.

        Note: The embedding is stored separately in the vector DB;
        this dict contains only scalar/string metadata fields.
        """
        return {
            "id": self.id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "last_accessed_at": self.last_accessed_at.isoformat(),
            "access_count": self.access_count,
            "source_type": self.source_type.value,
            "utility_score": self.utility_score,
            "goal_tags": ",".join(self.goal_tags),  # ChromaDB prefers flat values
            "decay_rate": self.decay_rate,
            "state": self.state.value,
            "consolidation_group": self.consolidation_group or "",
        }

    @classmethod
    def from_storage_dict(
        cls,
        data: dict[str, Any],
        embedding: list[float] | None = None,
    ) -> MemoryRecord:
        """
        Deserialize from a vector DB metadata dictionary.

        Args:
            data: Metadata dictionary from vector DB.
            embedding: Dense vector (loaded separately from vector DB).

        Returns:
            Reconstructed MemoryRecord.
        """
        goal_tags_raw = data.get("goal_tags", "")
        goal_tags = [t for t in goal_tags_raw.split(",") if t] if goal_tags_raw else []

        return cls(
            id=data["id"],
            content=data["content"],
            embedding=embedding.tolist() if hasattr(embedding, 'tolist') else (embedding if embedding is not None else []),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed_at=datetime.fromisoformat(data["last_accessed_at"]),
            access_count=int(data.get("access_count", 0)),
            source_type=SourceType(data.get("source_type", "conversation")),
            utility_score=float(data.get("utility_score", 0.5)),
            goal_tags=goal_tags,
            decay_rate=float(data.get("decay_rate", 0.05)),
            state=MemoryState(data.get("state", "active")),
            consolidation_group=data.get("consolidation_group") or None,
        )

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"MemoryRecord(id={self.id[:8]}..., "
            f"utility={self.utility_score:.2f}, "
            f"state={self.state.value}, "
            f"content='{content_preview}')"
        )
