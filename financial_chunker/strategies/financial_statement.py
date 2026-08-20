"""
Financial Statement Chunking Strategy.
Specialized for Income Statements, Balance Sheets, Cash Flow Statements, and Notes to Financial Statements.
"""

from typing import List
from .base import BaseChunkingStrategy
from ..types import Document, Chunk, ChunkType, DocumentType
from ..rules.table_processor import TableProcessor
from ..rules.footnote_handler import FootnoteHandler
from ..prefix.context_prefixer import ContextPrefixer
from ..hierarchy.tree_builder import HierarchyTreeBuilder


class FinancialStatementStrategy(BaseChunkingStrategy):
    """Chunking strategy optimized for tabular Financial Statements and accompanying Notes."""

    def chunk(self, doc: Document) -> List[Chunk]:
        raw_text = doc.content

        # 1. Footnote Extraction
        clean_text, footnotes = FootnoteHandler.extract_footnote_definitions(raw_text)

        # 2. Extract structured tables
        tables = TableProcessor.extract_markdown_tables(clean_text)

        chunks: List[Chunk] = []

        # Document Root Chunk
        doc_root = Chunk(
            doc_id=doc.doc_id,
            text=f"Financial Statements: {doc.title}",
            context_prefix=f"[Document: {doc.title} | Type: Financial Statement]",
            metadata=doc.metadata,
            level=0,
            chunk_type=ChunkType.HEADER,
            breadcrumbs=[doc.title]
        )
        chunks.append(doc_root)

        if not tables:
            # Fallback to prose handling if no tables detected
            prefix = ContextPrefixer.build_prefix(
                doc_title=doc.title,
                doc_type=DocumentType.FINANCIAL_STATEMENT,
                breadcrumbs=[doc.title]
            )
            parent = Chunk(
                doc_id=doc.doc_id,
                text=clean_text,
                context_prefix=prefix,
                metadata=doc.metadata,
                parent_id=doc_root.id,
                level=1,
                chunk_type=ChunkType.PROSE,
                token_count=HierarchyTreeBuilder.estimate_token_count(clean_text),
                breadcrumbs=[doc.title],
                footnotes=footnotes
            )
            doc_root.child_ids.append(parent.id)
            chunks.append(parent)

            leaves = HierarchyTreeBuilder.split_into_leaf_chunks(
                parent_chunk=parent,
                doc_title=doc.title,
                doc_type=DocumentType.FINANCIAL_STATEMENT
            )
            chunks.extend(leaves)
            return chunks

        # Process each financial table
        for idx, (raw_tbl_text, tbl_data) in enumerate(tables):
            table_title = tbl_data.caption if tbl_data.caption else f"Financial Statement {idx+1}"
            breadcrumbs = [doc.title, table_title]

            formatted_text = TableProcessor.format_table_chunk_text(
                tbl_data, context_title=table_title
            )

            # Re-attach footnotes if referenced
            formatted_text = FootnoteHandler.attach_footnotes_to_chunk_text(formatted_text, footnotes)

            prefix = ContextPrefixer.build_prefix(
                doc_title=doc.title,
                doc_type=DocumentType.FINANCIAL_STATEMENT,
                breadcrumbs=breadcrumbs,
                chunk_type=ChunkType.TABLE
            )

            tbl_chunk = Chunk(
                doc_id=doc.doc_id,
                text=formatted_text,
                context_prefix=prefix,
                metadata={"table_title": table_title, **doc.metadata},
                parent_id=doc_root.id,
                level=1,
                chunk_type=ChunkType.TABLE,
                token_count=HierarchyTreeBuilder.estimate_token_count(formatted_text),
                breadcrumbs=breadcrumbs,
                footnotes=footnotes
            )
            doc_root.child_ids.append(tbl_chunk.id)
            chunks.append(tbl_chunk)

            # Create leaf child representation
            leaf_chunk = Chunk(
                doc_id=doc.doc_id,
                text=formatted_text,
                context_prefix=prefix,
                metadata={"table_title": table_title, **doc.metadata},
                parent_id=tbl_chunk.id,
                level=2,
                chunk_type=ChunkType.TABLE,
                token_count=HierarchyTreeBuilder.estimate_token_count(formatted_text),
                breadcrumbs=breadcrumbs,
                footnotes=footnotes
            )
            tbl_chunk.child_ids.append(leaf_chunk.id)
            chunks.append(leaf_chunk)

        return chunks
