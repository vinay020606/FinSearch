"""
Dense Semantic Vector Store module.
Simulates dense vector embedding indexing and cosine similarity retrieval for Financial RAG.
"""

import math
from typing import List, Dict, Tuple
from financial_chunker.types import Chunk
from .types import RetrievalResult


class SimpleEmbeddingModel:
    """Lightweight deterministic model generating normalized dense semantic feature vectors."""

    @staticmethod
    def embed_text(text: str, dim: int = 128) -> List[float]:
        """Generates a normalized dim-dimensional embedding vector for input text."""
        vector = [0.0] * dim
        words = text.lower().split()
        if not words:
            return vector

        for word in words:
            # Deterministic feature hashing
            hash_val = hash(word)
            idx = abs(hash_val) % dim
            vector[idx] += 1.0 + (len(word) * 0.1)

        # Normalize vector to unit length
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


class InMemoryVectorStore:
    """In-Memory Dense Semantic Vector Store supporting cosine similarity retrieval."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.corpus_chunks: List[Chunk] = []
        self.vectors: List[List[float]] = []

    def index(self, chunks: List[Chunk]) -> None:
        """Embeds and indexes document chunks into the vector store."""
        self.corpus_chunks = chunks
        self.vectors = []
        for chunk in chunks:
            vector = SimpleEmbeddingModel.embed_text(chunk.embedding_text, dim=self.embedding_dim)
            self.vectors.append(vector)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Searches vector index using cosine similarity against query embedding."""
        if not self.corpus_chunks or not query:
            return []

        query_vector = SimpleEmbeddingModel.embed_text(query, dim=self.embedding_dim)
        scores: List[Tuple[int, float]] = []

        for idx, doc_vector in enumerate(self.vectors):
            # Cosine similarity (since vectors are unit length, dot product equals cosine similarity)
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
                retriever_name="VectorSearch"
            ))

        return results
