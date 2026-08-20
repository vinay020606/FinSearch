"""
End-to-end integration tests for HybridRetriever.
"""

from financial_chunker import Document, DocumentType
from financial_retrieval import HybridRetriever


def test_hybrid_retriever_end_to_end():
    doc1 = Document(
        doc_id="doc_10k",
        title="Acme Corp 2023 10-K",
        content="""
PART I
ITEM 1A. RISK FACTORS
We adopted ASC 606 revenue recognition standard in FY23.

ITEM 7. MD&A
Net income reached $50 million, up 15% YoY.
""",
        doc_type=DocumentType.SEC_FILING
    )

    retriever = HybridRetriever()
    indexed_count = retriever.index_documents([doc1])
    assert indexed_count > 0

    # Query matching exact accounting standard code
    results = retriever.search("ASC 606 revenue recognition", top_k=2)
    assert len(results) > 0
    top_result = results[0]
    assert "ASC 606" in top_result.summary_text
    assert top_result.bm25_rank is not None
    assert top_result.rrf_score > 0.0


test_hybrid_retriever_end_to_end()
print("HybridRetriever integration test passed!")
