"""
Example Usage script demonstrating the Production Financial Document Chunker.
Runs chunking on SEC 10-K, Earnings Call Transcripts, and Financial Statement Tables.
"""

from financial_chunker import (
    Document,
    DocumentType,
    ProductionFinancialChunker,
    ChunkType
)


def run_demo():
    chunker = ProductionFinancialChunker()

    print("=" * 80)
    print(" FINANCIAL DOCUMENT CHUNKER - PRODUCTION DEMO")
    print("=" * 80)

    # 1. SEC Filing 10-K Demo
    sec_doc_text = """
PART I

ITEM 1A. RISK FACTORS

Our business is subject to macroeconomic uncertainties. Global inflation and foreign currency fluctuations reduced operating margin by 150 bps to 22.5% in FY23. (1)

| Segment | Revenue FY23 | Revenue FY22 | YoY Growth |
| --- | --- | --- | --- |
| North America | $50.5B | $45.2B | +11.7% |
| Europe | $24.8B | $23.1B | +7.3% |
| Asia Pacific | $14.2B | $12.8B | +10.9% |

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION (MD&A)

Net Cash Provided by Operating Activities was $110.5 billion for FY23, compared to $122.2 billion for FY22. We completed $50 million in senior note repayments.

(1) Operating margin calculations exclude non-recurring restructuring charges of $12M.
"""
    sec_doc = Document(
        doc_id="sec_10k_2023",
        title="Apple Inc. 2023 Form 10-K",
        content=sec_doc_text,
        doc_type=DocumentType.SEC_FILING
    )

    print("\n[1] CHUNKING SEC 10-K FILING...")
    sec_chunks = chunker.chunk_document(sec_doc)
    print(f"--> Total Chunks Generated: {len(sec_chunks)}")

    for idx, chunk in enumerate(sec_chunks):
        if chunk.level == 0:
            continue
        print(f"\n--- Chunk #{idx} [Type: {chunk.chunk_type.value} | Level: {chunk.level}] ---")
        print(f"Parent ID: {chunk.parent_id}")
        print(f"Breadcrumbs: {' > '.join(chunk.breadcrumbs)}")
        print(f"Context Prefix: {chunk.context_prefix}")
        print("Embedding Text Sample:")
        print(chunk.embedding_text[:350] + ("..." if len(chunk.embedding_text) > 350 else ""))

    # 2. Earnings Call Transcript Demo
    transcript_text = """
PREPARED REMARKS

Tim Cook (CEO):
Good afternoon everyone. We are pleased to report revenue of $85.8 billion for our fiscal third quarter, up 5% from a year ago. iPhone revenue came in at $39.3 billion.

QUESTION AND ANSWER

Toni Sacconaghi (Sanford Bernstein):
Thank you. Could you comment on your capital return plan regarding $50 billion in share buybacks and gross margin expectations for Q4?

Luca Maestri (CFO):
Thanks Toni. We expect gross margin to be between 45.5% and 46.5%. We remain very committed to net cash neutral over time.
"""
    transcript_doc = Document(
        doc_id="transcript_q3_2024",
        title="Apple Inc. Q3 2024 Earnings Call",
        content=transcript_text,
        doc_type=DocumentType.EARNINGS_TRANSCRIPT
    )

    print("\n" + "=" * 80)
    print("[2] CHUNKING EARNINGS CALL TRANSCRIPT...")
    transcript_chunks = chunker.chunk_document(transcript_doc)
    print(f"--> Total Chunks Generated: {len(transcript_chunks)}")

    for idx, chunk in enumerate(transcript_chunks):
        if chunk.level == 0:
            continue
        print(f"\n--- Chunk #{idx} [Type: {chunk.chunk_type.value} | Level: {chunk.level}] ---")
        print(f"Speaker: {chunk.metadata.get('speaker', 'N/A')}")
        print(f"Context Prefix: {chunk.context_prefix}")
        print("Embedding Text:")
        print(chunk.embedding_text)

    print("\n" + "=" * 80)
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
