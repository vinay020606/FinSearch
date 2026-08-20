"""
Financial BM25 Engine with Custom Financial Tokenizer.
Tuned for exact matches on ASC accounting standard codes, CUSIPs, tickers, fiscal tags, and financial metrics.
"""

import math
import re
from typing import List, Dict, Set, Tuple
from financial_chunker.types import Chunk
from .types import RetrievalResult


# Tokenizer regex preserving accounting codes (ASC 606), numbers, currency symbols, hyphens, and tickers
FINANCIAL_TOKEN_REGEX = re.compile(
    r'\bASC\s*\d{3}\b|\b[A-Z]{1,5}\b|\bFY\d{2,4}\b|\bQ[1-4]\b|[$€£¥₹]?\d+(?:\.\d+)?(?:[mMbBkK]|%|bps)?|\w+(?:-\w+)*',
    re.IGNORECASE
)


class FinancialTokenizer:
    """Tokenizer preserving financial terms, standard codes, numbers, and identifiers."""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        if not text:
            return []
        
        # Normalize whitespace while preserving key financial codes
        tokens = []
        for match in FINANCIAL_TOKEN_REGEX.finditer(text):
            tok = match.group(0).lower().strip()
            if tok and len(tok) > 1 or tok.isdigit() or tok in ['$', '€', '£']:
                tokens.append(tok)
        return tokens


class FinancialBM25:
    """
    Okapi BM25 implementation tuned for financial text retrieval.
    Default parameters: k1=1.2 (saturation), b=0.75 (length normalization).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_chunks: List[Chunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.total_docs: int = 0

    def index(self, chunks: List[Chunk]) -> None:
        """Indexes document chunks into the BM25 sparse keyword store."""
        self.corpus_chunks = chunks
        self.total_docs = len(chunks)
        self.doc_tokens = []
        self.doc_lengths = []
        self.doc_freqs = []

        total_length = 0
        df_counter: Dict[str, int] = {}

        for chunk in chunks:
            tokens = FinancialTokenizer.tokenize(chunk.embedding_text)
            self.doc_tokens.append(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)

            # Unique terms in doc for IDF
            for t in set(tokens):
                df_counter[t] = df_counter.get(t, 0) + 1

        self.avgdl = (total_length / self.total_docs) if self.total_docs > 0 else 1.0

        # Calculate Okapi BM25 IDF
        self.idf = {}
        for term, df in df_counter.items():
            # BM25 positive IDF calculation formula
            self.idf[term] = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Searches the BM25 index for a given query string."""
        if not self.corpus_chunks or not query:
            return []

        query_tokens = FinancialTokenizer.tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []

        for idx in range(self.total_docs):
            doc_len = self.doc_lengths[idx]
            freqs = self.doc_freqs[idx]
            score = 0.0

            for q_term in query_tokens:
                if q_term not in freqs:
                    continue
                f = freqs[q_term]
                idf_val = self.idf.get(q_term, 0.0)
                
                # BM25 term score calculation
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += idf_val * (numerator / denominator)

            if score > 0.0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(scores[:top_k], start=1):
            results.append(RetrievalResult(
                chunk_id=self.corpus_chunks[idx].id,
                score=score,
                rank=rank,
                chunk=self.corpus_chunks[idx],
                retriever_name="BM25"
            ))

        return results
