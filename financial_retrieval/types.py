"""
Type definitions and data models for Financial Hybrid Search and RRF Fusion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from financial_chunker.types import Chunk


@dataclass
class RetrievalResult:
    """Individual retrieval candidate result from a single retriever (Dense or BM25)."""
    chunk_id: str
    score: float
    rank: int
    chunk: Chunk
    retriever_name: str


@dataclass
class HybridResult:
    """Fused search result combining vector semantic search and BM25 exact keyword match."""
    chunk_id: str
    rrf_score: float
    vector_rank: Optional[int]
    vector_score: Optional[float]
    bm25_rank: Optional[int]
    bm25_score: Optional[float]
    chunk: Chunk
    parent_chunk_text: Optional[str] = None
    breadcrumbs: List[str] = field(default_factory=list)

    @property
    def summary_text(self) -> str:
        """Formatted summary text showing contextual prefix and chunk content."""
        return self.chunk.embedding_text
