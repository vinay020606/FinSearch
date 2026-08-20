"""
End-to-End integration tests for ProductionFinancialChunker.
"""

from financial_chunker import Document, DocumentType, ProductionFinancialChunker, ChunkType


def test_sec_filing_end_to_end():
    sec_content = """
PART I

ITEM 1A. RISK FACTORS

Global economic conditions may adversely affect our revenue of $50 million. (1)

| Fiscal Year | Revenue | Net Income |
| --- | --- | --- |
| 2023 | $50.0M | $12.5M |

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION

We experienced strong growth in liquidity, achieving $15.0 billion in operating cash flow.

(1) Excludes stock-based compensation.
"""
    doc = Document(
        doc_id="sec_001",
        title="Acme Corp 2023 10-K",
        content=sec_content,
        doc_type=DocumentType.SEC_FILING
    )

    chunker = ProductionFinancialChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0

    # Check contextual prefix on embedding_text
    leaf_chunks = [c for c in chunks if c.level > 0]
    assert any("[Document: Acme Corp 2023 10-K" in c.embedding_text for c in leaf_chunks)

    # Check hierarchy
    root_chunk = [c for c in chunks if c.level == 0][0]
    assert len(root_chunk.child_ids) > 0

    # Check table chunk presence
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
    assert len(table_chunks) > 0

    # Check footnote retention
    footnote_attached_chunks = [c for c in chunks if "Footnotes" in c.text]
    assert len(footnote_attached_chunks) > 0


def test_earnings_transcript_end_to_end():
    transcript_content = """
PREPARED REMARKS

Tim Cook (CEO):
Good afternoon. In Q3 2024, our revenue reached $85.8 billion, up 5% YoY.

QUESTION AND ANSWER

Toni Sacconaghi (Sanford Bernstein):
Can you speak about your capital allocation strategy regarding $50 billion in share buybacks?

Tim Cook (CEO):
Thank you Toni. We remain committed to returning value to shareholders while investing in R&D.
"""
    doc = Document(
        doc_id="call_001",
        title="Apple Q3 2024 Earnings Call",
        content=transcript_content,
        doc_type=DocumentType.EARNINGS_TRANSCRIPT
    )

    chunker = ProductionFinancialChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 0
    speaker_chunks = [c for c in chunks if "Tim Cook" in c.text]
    assert len(speaker_chunks) > 0
    assert any("Speaker: Tim Cook" in c.context_prefix for c in speaker_chunks)


test_sec_filing_end_to_end()
test_earnings_transcript_end_to_end()
print("All End-to-End integration tests passed successfully!")
