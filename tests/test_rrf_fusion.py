"""
Unit tests for RRFFusion.
"""

from financial_chunker import Chunk
from financial_retrieval.types import RetrievalResult
from financial_retrieval.rrf_fusion import RRFFusion


def test_rrf_fusion_scoring():
    c1 = Chunk(id="chunk_1", text="Segment Revenue for North America $50M.")
    c2 = Chunk(id="chunk_2", text="Operating Margin expanded by 150 bps.")

    vec_results = [
        RetrievalResult(chunk_id="chunk_1", score=0.95, rank=1, chunk=c1, retriever_name="Vector"),
        RetrievalResult(chunk_id="chunk_2", score=0.80, rank=2, chunk=c2, retriever_name="Vector")
    ]

    bm25_results = [
        RetrievalResult(chunk_id="chunk_2", score=5.2, rank=1, chunk=c2, retriever_name="BM25"),
        RetrievalResult(chunk_id="chunk_1", score=2.1, rank=2, chunk=c1, retriever_name="BM25")
    ]

    fused = RRFFusion.fuse(vector_results=vec_results, bm25_results=bm25_results, k=60.0)
    assert len(fused) == 2
    # Both chunks appear in rank 1 and 2 across different retrievers, so their RRF scores equal 1/61 + 1/62
    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert abs(fused[0].rrf_score - expected_score) < 1e-5


test_rrf_fusion_scoring()
print("RRFFusion tests passed!")
