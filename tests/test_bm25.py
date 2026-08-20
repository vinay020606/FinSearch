"""
Unit tests for FinancialBM25 and FinancialTokenizer.
"""

from financial_chunker import Chunk
from financial_retrieval.bm25 import FinancialBM25, FinancialTokenizer


def test_tokenizer_preserves_financial_terms():
    raw_text = "Revenue under ASC 606 was $50.5M for AAPL in FY23."
    tokens = FinancialTokenizer.tokenize(raw_text)
    assert "asc 606" in tokens or ("asc" in tokens and "606" in tokens)
    assert "$50.5m" in tokens or "50.5m" in tokens
    assert "aapl" in tokens
    assert "fy23" in tokens


def test_bm25_exact_keyword_retrieval():
    c1 = Chunk(id="c1", text="Company adopted ASC 606 revenue recognition guidelines in 2023.", context_prefix="[Doc: 10-K]")
    c2 = Chunk(id="c2", text="Capital expenditure was $10 billion for corporate facilities.", context_prefix="[Doc: 10-K]")
    c3 = Chunk(id="c3", text="ASC 842 lease accounting standards impact total assets.", context_prefix="[Doc: 10-K]")

    bm25 = FinancialBM25()
    bm25.index([c1, c2, c3])

    results = bm25.search("ASC 606 revenue recognition", top_k=2)
    assert len(results) > 0
    assert results[0].chunk_id == "c1"


test_tokenizer_preserves_financial_terms()
test_bm25_exact_keyword_retrieval()
print("FinancialBM25 tests passed!")
