"""
Generic Financial Prose Chunking Strategy.
Handles general financial research reports, analyst notes, and unstructured financial prose.
"""

import re
from typing import List, Tuple, Dict, Any
from .base import BaseChunkingStrategy
from ..types import Document, Chunk, ChunkType, DocumentType
from ..rules.table_processor import TableProcessor
from ..rules.footnote_handler import FootnoteHandler
from ..rules.number_unit_protector import NumberUnitProtector
from ..prefix.context_prefixer import ContextPrefixer
from ..hierarchy.tree_builder import HierarchyTreeBuilder

# Regex for Markdown headings: # Heading 1, ## Heading 2, ### Heading 3
HEADING_REGEX = re.compile(
    r'(?:\n|^)\s*(#{1,4})\s+(.+)$',
    re.MULTILINE
)


class GenericProseStrategy(BaseChunkingStrategy):
    """Chunking strategy optimized for general financial research notes and prose documents."""

    def chunk(self, doc: Document) -> List[Chunk]:
        raw_text = doc.content

        # 1. Footnote extraction
        clean_text, footnotes = FootnoteHandler.extract_footnote_definitions(raw_text)

        # 2. Table extraction & isolation
        tables = TableProcessor.extract_markdown_tables(clean_text)

        table_placeholders: Dict[str, Tuple[str, Any]] = {}
        processed_text = clean_text

        for idx, (raw_tbl_text, tbl_data) in enumerate(tables):
            placeholder = f"__GENERIC_TABLE_PLACEHOLDER_{idx}__"
            table_placeholders[placeholder] = (raw_tbl_text, tbl_data)
            processed_text = processed_text.replace(raw_tbl_text, f"\n\n{placeholder}\n\n", 1)

        chunks: List[Chunk] = []

        doc_root = Chunk(
            doc_id=doc.doc_id,
            text=f"Document: {doc.title}",
            context_prefix=f"[Document: {doc.title}]",
            metadata=doc.metadata,
            level=0,
            chunk_type=ChunkType.HEADER,
            breadcrumbs=[doc.title]
        )
        chunks.append(doc_root)

        # 3. Parse Markdown headings
        sections = self._split_by_headings(processed_text)

        for heading_path, section_body in sections:
            breadcrumbs = [doc.title] + heading_path if heading_path else [doc.title, "Overview"]

            prefix = ContextPrefixer.build_prefix(
                doc_title=doc.title,
                doc_type=DocumentType.GENERIC_PROSE,
                breadcrumbs=breadcrumbs
            )

            parent_chunk = Chunk(
                doc_id=doc.doc_id,
                text=section_body,
                context_prefix=prefix,
                metadata={"heading": breadcrumbs[-1], **doc.metadata},
                parent_id=doc_root.id,
                level=1,
                chunk_type=ChunkType.PROSE,
                token_count=HierarchyTreeBuilder.estimate_token_count(section_body),
                breadcrumbs=breadcrumbs,
                footnotes=footnotes
            )
            doc_root.child_ids.append(parent_chunk.id)
            chunks.append(parent_chunk)

            # Re-insert tables or process prose blocks
            paragraphs = section_body.split('\n\n')
            for p in paragraphs:
                p_str = p.strip()
                if not p_str:
                    continue

                if p_str in table_placeholders:
                    raw_tbl, tbl_data = table_placeholders[p_str]
                    fmt_tbl = TableProcessor.format_table_chunk_text(tbl_data, " > ".join(breadcrumbs))
                    tbl_prefix = ContextPrefixer.build_prefix(
                        doc_title=doc.title,
                        doc_type=DocumentType.GENERIC_PROSE,
                        breadcrumbs=breadcrumbs,
                        chunk_type=ChunkType.TABLE
                    )
                    tbl_chunk = Chunk(
                        doc_id=doc.doc_id,
                        text=fmt_tbl,
                        context_prefix=tbl_prefix,
                        metadata=doc.metadata,
                        parent_id=parent_chunk.id,
                        level=2,
                        chunk_type=ChunkType.TABLE,
                        token_count=HierarchyTreeBuilder.estimate_token_count(fmt_tbl),
                        breadcrumbs=breadcrumbs,
                        footnotes=footnotes
                    )
                    parent_chunk.child_ids.append(tbl_chunk.id)
                    chunks.append(tbl_chunk)
                else:
                    protected_p = NumberUnitProtector.protect(p_str)
                    temp_parent = Chunk(
                        doc_id=doc.doc_id,
                        text=protected_p,
                        context_prefix=prefix,
                        metadata=doc.metadata,
                        id=parent_chunk.id,
                        level=1,
                        breadcrumbs=breadcrumbs,
                        footnotes=footnotes
                    )
                    leaves = HierarchyTreeBuilder.split_into_leaf_chunks(
                        parent_chunk=temp_parent,
                        doc_title=doc.title,
                        doc_type=DocumentType.GENERIC_PROSE
                    )
                    for leaf in leaves:
                        leaf.text = FootnoteHandler.attach_footnotes_to_chunk_text(leaf.text, footnotes)
                        chunks.append(leaf)

        return chunks

    def _split_by_headings(self, text: str) -> List[Tuple[List[str], str]]:
        """Splits document text by markdown headings (#, ##, ###) into (heading_path, body) pairs."""
        matches = list(HEADING_REGEX.finditer(text))
        if not matches:
            return [(["Overview"], text)]

        sections = []
        heading_stack: List[Tuple[int, str]] = []

        for i in range(len(matches)):
            match = matches[i]
            level = len(match.group(1))
            title = match.group(2).strip()

            # Maintain active heading hierarchy stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            current_path = [h[1] for h in heading_stack]

            start_idx = match.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            body = text[start_idx:end_idx].strip()

            sections.append((current_path, body))

        return sections
