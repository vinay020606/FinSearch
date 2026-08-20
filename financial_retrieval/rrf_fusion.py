"""
Reciprocal Rank Fusion (RRF) module.
Combines semantic vector rankings and lexical BM25 rankings into a unified hybrid score.
"""

from typing import List, Dict, Tuple, Optional
from financial_chunker.types import Chunk
from .types import RetrievalResult, HybridResult


class RRFFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) algorithm.
    RRF Score(d) = sum_{m in M} ( w_m / (k + r_m(d)) )
    Default k=60 prevents high-ranking outliers from dominating fusion scores.
    """

    @staticmethod
    def fuse(
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: float = 60.0,
        weight_vector: float = 1.0,
        weight_bm25: float = 1.0,
        parent_chunk_map: Optional[Dict[str, Chunk]] = None
    ) -> List[HybridResult]:
        """
        Fuses ranked candidate lists from Vector Search and BM25 Search using Reciprocal Rank Fusion.
        """
        fused_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        vec_ranks: Dict[str, int] = {}
        vec_scores: Dict[str, float] = {}
        bm25_ranks: Dict[str, int] = {}
        bm25_scores: Dict[str, float] = {}

        # 1. Process Vector Search results
        for item in vector_results:
            cid = item.chunk_id
            chunk_map[cid] = item.chunk
            vec_ranks[cid] = item.rank
            vec_scores[cid] = item.score

            rrf_contrib = weight_vector / (k + item.rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # 2. Process BM25 Keyword Search results
        for item in bm25_results:
            cid = item.chunk_id
            chunk_map[cid] = item.chunk
            bm25_ranks[cid] = item.rank
            bm25_scores[cid] = item.score

            rrf_contrib = weight_bm25 / (k + item.rank)
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # 3. Sort fused candidates by RRF score descending
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results: List[HybridResult] = []
        for cid, rrf_score in sorted_candidates:
            chunk = chunk_map[cid]
            parent_text = None

            if parent_chunk_map and chunk.parent_id and chunk.parent_id in parent_chunk_map:
                parent_chunk_text = parent_chunk_map[chunk.parent_id].text
            else:
                parent_chunk_text = None

            results.append(HybridResult(
                chunk_id=cid,
                rrf_score=rrf_score,
                vector_rank=vec_ranks.get(cid),
                vector_score=vec_scores.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                bm25_score=bm25_scores.get(cid),
                chunk=chunk,
                parent_chunk_text=parent_chunk_text,
                breadcrumbs=chunk.breadcrumbs
            ))

        return results
