"""
Main Orchestrator module for Production Financial Chunking.
"""

import re
from typing import List, Dict, Type, Optional
from .types import Document, Chunk, DocumentType
from .strategies.base import BaseChunkingStrategy
from .strategies.sec_filing import SecFilingStrategy
from .strategies.earnings_transcript import EarningsTranscriptStrategy
from .strategies.financial_statement import FinancialStatementStrategy
from .strategies.generic_prose import GenericProseStrategy


class ProductionFinancialChunker:
    """
    Production-grade Financial Document Chunker orchestrator.
    Auto-detects document types or executes specific structural chunking strategies.
    """

    def __init__(self):
        self._strategies: Dict[DocumentType, BaseChunkingStrategy] = {
            DocumentType.SEC_FILING: SecFilingStrategy(),
            DocumentType.EARNINGS_TRANSCRIPT: EarningsTranscriptStrategy(),
            DocumentType.FINANCIAL_STATEMENT: FinancialStatementStrategy(),
            DocumentType.GENERIC_PROSE: GenericProseStrategy(),
        }

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Main entry point for chunking a financial document.
        
        Args:
            doc: Input document with raw content, metadata, and optional document type.
            
        Returns:
            List of production-ready Chunk objects with context prefixes, hierarchical metadata, and parent-child links.
        """
        doc_type = doc.doc_type

        if doc_type == DocumentType.AUTO:
            doc_type = self.detect_document_type(doc.content)
            doc.doc_type = doc_type

        strategy = self._strategies.get(doc_type, self._strategies[DocumentType.GENERIC_PROSE])
        return strategy.chunk(doc)

    def detect_document_type(self, text: str) -> DocumentType:
        """
        Auto-detects financial document type using structural heuristic patterns.
        """
        text_upper = text.upper()

        # 1. SEC Filings (10-K, 10-Q, 8-K)
        sec_keywords = ["ITEM 1A.", "ITEM 7.", "PART I", "FORM 10-K", "FORM 10-Q", "SECURITIES AND EXCHANGE COMMISSION"]
        sec_matches = sum(1 for kw in sec_keywords if kw in text_upper)
        if sec_matches >= 2 or re.search(r'\bITEM\s+\d+[A-Z]?\b', text_upper):
            return DocumentType.SEC_FILING

        # 2. Earnings Call Transcripts
        transcript_keywords = ["PREPARED REMARKS", "QUESTION AND ANSWER", "OPERATOR:", "EXECUTIVE PRESENTATION", "Q&A SESSION"]
        transcript_matches = sum(1 for kw in transcript_keywords if kw in text_upper)
        if transcript_matches >= 2 or re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s*\([^)]+\):', text):
            return DocumentType.EARNINGS_TRANSCRIPT

        # 3. Financial Statements / Tables
        statement_keywords = ["CONSOLIDATED STATEMENTS OF OPERATIONS", "BALANCE SHEETS", "CASH FLOWS", "STATEMENT OF FINANCIAL POSITION"]
        if any(kw in text_upper for kw in statement_keywords):
            return DocumentType.FINANCIAL_STATEMENT

        return DocumentType.GENERIC_PROSE
