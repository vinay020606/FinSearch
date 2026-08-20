"""
Type definitions and data models for the Financial Document Chunker.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid


class DocumentType(str, Enum):
    SEC_FILING = "sec_filing"            # 10-K, 10-Q, 8-K filings with Item/Part structures
    EARNINGS_TRANSCRIPT = "earnings_transcript" # Earnings calls with speaker turns & Q&A
    FINANCIAL_STATEMENT = "financial_statement" # Balance Sheet, Income Statement, Cash Flow
    GENERIC_PROSE = "generic_prose"      # General financial reports & analyst notes
    AUTO = "auto"                        # Auto-detect document type


class ChunkType(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    HEADER = "header"
    QA_PAIR = "qa_pair"
    FOOTNOTE = "footnote"
    EXECUTIVE_REMARKS = "executive_remarks"


@dataclass
class Footnote:
    marker: str
    text: str
    symbol: Optional[str] = None


@dataclass
class TableData:
    caption: str
    headers: List[str]
    rows: List[List[str]]
    notes: List[str] = field(default_factory=list)
    raw_markdown: str = ""
    narrative_summary: str = ""


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    text: str = ""
    context_prefix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    level: int = 0  # 0=Document root, 1=Major Section, 2=Subsection, 3=Leaf Child Chunk
    chunk_type: ChunkType = ChunkType.PROSE
    token_count: int = 0
    breadcrumbs: List[str] = field(default_factory=list)
    footnotes: List[Footnote] = field(default_factory=list)

    @property
    def embedding_text(self) -> str:
        """Text used for vector embedding generation, incorporating contextual prefix."""
        if self.context_prefix:
            return f"{self.context_prefix.strip()}\n\n{self.text.strip()}"
        return self.text.strip()


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    doc_type: DocumentType = DocumentType.AUTO
    metadata: Dict[str, Any] = field(default_factory=dict)
