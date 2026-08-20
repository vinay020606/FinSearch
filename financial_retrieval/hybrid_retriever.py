"""
Hybrid Retriever Orchestrator module.
Integrates ProductionFinancialChunker, FinancialBM25, InMemoryVectorStore, and RRFFusion.
"""

from typing import List, Dict, Optional
from financial_chunker import Document, Chunk, ProductionFinancialChunker
from .types import HybridResult
from .bm25 import FinancialBM25
from .vector_store import InMemoryVectorStore
from .rrf_fusion import RRFFusion


class HybridRetriever:
    """
    Production Financial Hybrid Retriever.
    Combines dense semantic vector search and sparse keyword BM25 search using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, bm25_k1: float = 1.2, bm25_b: float = 0.75, rrf_k: float = 60.0):
        self.chunker = ProductionFinancialChunker()
        self.bm25 = FinancialBM25(k1=bm25_k1, b=bm25_b)
        self.vector_store = InMemoryVectorStore()
        self.rrf_k = rrf_k
        self.indexed_chunks: List[Chunk] = []
        self.parent_chunk_map: Dict[str, Chunk] = {}

    def index_documents(self, docs: List[Document]) -> int:
        """
        Processes financial documents into structured chunks and indexes them across both vector and BM25 engines.
        
        Args:
            docs: List of Document objects to chunk and index.
            
        Returns:
            Total number of indexed leaf/retrieval chunks.
        """
        all_chunks: List[Chunk] = []

        for doc in docs:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        # Build parent chunk map for context expansion
        self.parent_chunk_map = {c.id: c for c in all_chunks if c.child_ids}

        # Filter leaf retrieval chunks (level > 0)
        retrieval_chunks = [c for c in all_chunks if c.level > 0]
        self.indexed_chunks = retrieval_chunks

        # Index in both BM25 and Vector Store
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
        
        Args:
            query: Financial search query string.
            top_k: Number of final hybrid results to return.
            candidate_k: Number of top candidate results to fetch from each retriever before fusion.
            weight_vector: RRF multiplier weight for vector search.
            weight_bm25: RRF multiplier weight for BM25 search.
            
        Returns:
            List of HybridResult objects ordered by RRF fusion score.
        """
        if not self.indexed_chunks or not query:
            return []

        # 1. Fetch dense vector candidates
        vector_candidates = self.vector_store.search(query, top_k=candidate_k)

        # 2. Fetch sparse BM25 candidates
        bm25_candidates = self.bm25.search(query, top_k=candidate_k)

        # 3. Fuse candidate rankings using Reciprocal Rank Fusion (RRF)
        fused_results = RRFFusion.fuse(
            vector_results=vector_candidates,
            bm25_results=bm25_candidates,
            k=self.rrf_k,
            weight_vector=weight_vector,
            weight_bm25=weight_bm25,
            parent_chunk_map=self.parent_chunk_map
        )

        return fused_results[:top_k]
