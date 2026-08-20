"""
Dense Semantic Vector Store module.
Supports pluggable BaseEmbeddingModel providers (bge-large, cohere embed-v3, e5-mistral, etc.).
"""

from typing import List, Dict, Tuple, Optional
from financial_chunker.types import Chunk
from .types import RetrievalResult
from .embeddings.base import BaseEmbeddingModel
from .embeddings.providers import FallbackHashEmbedding


class InMemoryVectorStore:
    """In-Memory Dense Semantic Vector Store supporting pluggable embedding providers."""

    def __init__(self, embedding_model: Optional[BaseEmbeddingModel] = None):
        self.embedding_model = embedding_model or FallbackHashEmbedding(dim=1024)
        self.corpus_chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def index(self, chunks: List[Chunk]) -> None:
        """Embeds and indexes document chunks into the vector store."""
        self.corpus_chunks = chunks
        if not chunks:
            self.vectors = []
            return

        texts = [c.embedding_text for c in chunks]
        self.vectors = self.embedding_model.embed_documents(texts)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Searches vector index using cosine similarity against query embedding."""
        if not self.corpus_chunks or not query or not self.vectors:
            return []

        query_vector = self.embedding_model.embed_query(query)
        scores: List[Tuple[int, float]] = []

        for idx, doc_vector in enumerate(self.vectors):
            score = sum(q * d for q, d in zip(query_vector, doc_vector))
            if score > 0.0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            results.append(RetrievalResult(
                chunk_id=self.corpus_chunks[idx].id,
                score=score,
                rank=rank,
                chunk=self.corpus_chunks[idx],
                retriever_name=f"VectorSearch({self.embedding_model.model_name})"
            ))

        return results
