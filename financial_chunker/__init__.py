"""
Production Financial Document Chunker Package.
"""

from .types import Document, Chunk, DocumentType, ChunkType, TableData, Footnote
from .chunker import ProductionFinancialChunker
from .rules.number_unit_protector import NumberUnitProtector
from .rules.footnote_handler import FootnoteHandler
from .rules.table_processor import TableProcessor
from .prefix.context_prefixer import ContextPrefixer
from .hierarchy.tree_builder import HierarchyTreeBuilder

__all__ = [
    "Document",
    "Chunk",
    "DocumentType",
    "ChunkType",
    "TableData",
    "Footnote",
    "ProductionFinancialChunker",
    "NumberUnitProtector",
    "FootnoteHandler",
    "TableProcessor",
    "ContextPrefixer",
    "HierarchyTreeBuilder"
]
