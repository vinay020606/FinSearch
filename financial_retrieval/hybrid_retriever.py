"""
Hybrid Retriever Orchestrator module.
Integrates ProductionFinancialChunker, FinancialBM25, InMemoryVectorStore (with pluggable BaseEmbeddingModel), and RRFFusion.
"""

from typing import List, Dict, Optional
from financial_chunker import Document, Chunk, ProductionFinancialChunker
from .types import HybridResult
from .bm25 import FinancialBM25
from .vector_store import InMemoryVectorStore
from .rrf_fusion import RRFFusion
from .embeddings.base import BaseEmbeddingModel


class HybridRetriever:
    """
    Production Financial Hybrid Retriever.
    Combines dense semantic vector search and sparse keyword BM25 search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
        rrf_k: float = 60.0
    ):
        self.chunker = ProductionFinancialChunker()
        self.bm25 = FinancialBM25(k1=bm25_k1, b=bm25_b)
        self.vector_store = InMemoryVectorStore(embedding_model=embedding_model)
        self.rrf_k = rrf_k
        self.indexed_chunks: List[Chunk] = []
        self.parent_chunk_map: Dict[str, Chunk] = {}

    def index_documents(self, docs: List[Document]) -> int:
        """
        Processes financial documents into structured chunks and indexes them across both vector and BM25 engines.
        """
        all_chunks: List[Chunk] = []

        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        self.parent_chunk_map = {c.id: c for c in all_chunks if c.child_ids}

        retrieval_chunks = [c for c in all_chunks if c.level > 0]
        self.indexed_chunks = retrieval_chunks

        self.bm25.index(retrieval_chunks)
        self.vector_store.index(retrieval_chunks)

        return len(retrieval_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        weight_vector: float = 1.0,
        weight_bm25: float = 1.0
    ) -> List[HybridResult]:
        """
        Executes hybrid search across vector and BM25 indices and fuses candidate results using RRF.
        """
        if not self.indexed_chunks or not query:
            return []

        vector_candidates = self.vector_store.search(query, top_k=candidate_k)
        bm25_candidates = self.bm25.search(query, top_k=candidate_k)

        fused_results = RRFFusion.fuse(
            vector_results=vector_candidates,
            bm25_results=bm25_candidates,
            k=self.rrf_k,
            weight_vector=weight_vector,
            weight_bm25=weight_bm25,
            parent_chunk_map=self.parent_chunk_map
        )

        return fused_results[:top_k]
