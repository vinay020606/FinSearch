"""
Example Usage script demonstrating Financial Hybrid Search (BM25 + Vector Search + RRF Fusion).
"""

from financial_chunker import Document, DocumentType
from financial_retrieval import HybridRetriever


def run_hybrid_demo():
    print("=" * 80)
    print(" FINANCIAL HYBRID SEARCH DEMO (BM25 + VECTOR + RRF FUSION)")
    print("=" * 80)

    # 1. Prepare sample financial documents
    doc1 = Document(
        doc_id="sec_10k_aapl",
        title="Apple Inc. 2023 Form 10-K",
        content="""
PART I
ITEM 1A. RISK FACTORS
We adopted ASC 606 revenue recognition guidelines in FY23. Foreign currency fluctuations reduced operating margin by 150 bps to 22.5%. (1)

| Segment | Revenue FY23 | Revenue FY22 |
| --- | --- | --- |
| North America | $50.5B | $45.2B |
| Europe | $24.8B | $23.1B |

ITEM 7. MD&A
Net Cash Provided by Operating Activities was $110.5 billion for FY23, compared to $122.2 billion for FY22.

(1) Excludes stock-based compensation of $12M.
""",
        doc_type=DocumentType.SEC_FILING
    )

    doc2 = Document(
        doc_id="transcript_q3_aapl",
        title="Apple Inc. Q3 2024 Earnings Call",
        content="""
PREPARED REMARKS

Tim Cook (CEO):
Good afternoon. In Q3 2024, our revenue reached $85.8 billion, up 5% YoY. iPhone revenue was $39.3 billion.

QUESTION AND ANSWER

Toni Sacconaghi (Sanford Bernstein):
Could you comment on your $50 billion share buyback plan and gross margin expectations?

Luca Maestri (CFO):
We expect gross margin between 45.5% and 46.5%.
""",
        doc_type=DocumentType.EARNINGS_TRANSCRIPT
    )

    # 2. Instantiate and Index Documents into Hybrid Retriever
    retriever = HybridRetriever(bm25_k1=1.2, bm25_b=0.75, rrf_k=60.0)
    print("\n--> Chunking and indexing financial documents...")
    indexed_count = retriever.index_documents([doc1, doc2])
    print(f"--> Total Retrieval Chunks Indexed: {indexed_count}")

    # 3. Test Queries
    queries = [
        "ASC 606 revenue recognition standard",
        "Tim Cook capital return $50 billion share buyback",
        "North America segment revenue FY23"
    ]

    for q in queries:
        print("\n" + "-" * 80)
        print(f" QUERY: '{q}'")
        print("-" * 80)

        results = retriever.search(q, top_k=2)

        for rank, res in enumerate(results, start=1):
            print(f"\n[Rank #{rank}] RRF Score: {res.rrf_score:.5f}")
            print(f"Vector Rank: {res.vector_rank} (Score: {res.vector_score:.4f} if res.vector_score else N/A)")
            print(f"BM25 Rank: {res.bm25_rank} (Score: {res.bm25_score:.4f} if res.bm25_score else N/A)")
            print(f"Breadcrumbs: {' > '.join(res.breadcrumbs)}")
            print("Retrieved Text Preview:")
            print(res.summary_text[:300] + ("..." if len(res.summary_text) > 300 else ""))

    print("\n" + "=" * 80)
    print(" HYBRID RETRIEVAL DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_hybrid_demo()
