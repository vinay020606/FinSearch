"""
SEC Filing Chunking Strategy.
Specialized for 10-K, 10-Q, 8-K filings with Item/Part section headers and regulatory structure.
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

# Regex matching SEC headings strictly on a single line
SEC_HEADER_REGEX = re.compile(
    r'(?:^|\n)\s*((?:PART\s+[I|V|X]+|ITEM\s+\d+[A-Z]?(?:[\.\:\s][^\n]+)?))(?=\n|$)',
    re.IGNORECASE
)

PART_REGEX = re.compile(r'^\s*PART\s+[I|V|X]+', re.IGNORECASE)
ITEM_REGEX = re.compile(r'^\s*ITEM\s+\d+[A-Z]?', re.IGNORECASE)


class SecFilingStrategy(BaseChunkingStrategy):
    """Chunking strategy optimized for SEC regulatory filings (10-K, 10-Q, 8-K)."""

    def chunk(self, doc: Document) -> List[Chunk]:
        raw_text = doc.content

        # 1. Extract footnotes
        clean_text, footnotes = FootnoteHandler.extract_footnote_definitions(raw_text)

        # 2. Extract tables and replace with placeholders
        extracted_tables = TableProcessor.extract_markdown_tables(clean_text)

        table_placeholders: Dict[str, Tuple[str, Any]] = {}
        processed_text = clean_text

        for idx, (raw_tbl_text, tbl_data) in enumerate(extracted_tables):
            placeholder = f"__SEC_TABLE_PLACEHOLDER_{idx}__"
            table_placeholders[placeholder] = (raw_tbl_text, tbl_data)
            processed_text = processed_text.replace(raw_tbl_text, f"\n\n{placeholder}\n\n", 1)

        # 3. Split along SEC Part and Item headings
        section_blocks = self._split_by_sec_headings(processed_text)

        chunks: List[Chunk] = []
        doc_root_chunk = Chunk(
            doc_id=doc.doc_id,
            text=f"Document Root: {doc.title}",
            context_prefix=f"[Document: {doc.title} | Type: SEC Filing]",
            metadata=doc.metadata,
            level=0,
            chunk_type=ChunkType.HEADER,
            breadcrumbs=[doc.title]
        )
        chunks.append(doc_root_chunk)

        current_part = ""
        current_item = ""

        for heading, section_body in section_blocks:
            clean_heading = heading.strip()
            if not clean_heading and not section_body:
                continue

            if PART_REGEX.match(clean_heading):
                current_part = clean_heading
                current_item = ""
            elif ITEM_REGEX.match(clean_heading):
                current_item = clean_heading

            breadcrumbs = [doc.title]
            if current_part:
                breadcrumbs.append(current_part)
            if current_item and current_item != current_part:
                breadcrumbs.append(current_item)
            if not current_part and not current_item and clean_heading:
                breadcrumbs.append(clean_heading)

            # Extract table placeholders vs prose text inside section_body
            paragraphs = section_body.split('\n\n')
            section_elements = []

            for p in paragraphs:
                p_str = p.strip()
                if not p_str:
                    continue
                if p_str in table_placeholders:
                    raw_tbl_text, tbl_data = table_placeholders[p_str]
                    section_elements.append(('table', tbl_data))
                else:
                    section_elements.append(('prose', p_str))

            # Build parent section chunk
            prefix = ContextPrefixer.build_prefix(
                doc_title=doc.title,
                doc_type=DocumentType.SEC_FILING,
                breadcrumbs=breadcrumbs
            )

            parent_chunk = Chunk(
                doc_id=doc.doc_id,
                text=section_body,
                context_prefix=prefix,
                metadata={"heading": clean_heading, **doc.metadata},
                parent_id=doc_root_chunk.id,
                level=1,
                chunk_type=ChunkType.PROSE,
                token_count=HierarchyTreeBuilder.estimate_token_count(section_body),
                breadcrumbs=breadcrumbs,
                footnotes=footnotes
            )
            doc_root_chunk.child_ids.append(parent_chunk.id)
            chunks.append(parent_chunk)

            # Build leaf child chunks
            for elem_type, elem_content in section_elements:
                if elem_type == 'table':
                    formatted_tbl_text = TableProcessor.format_table_chunk_text(
                        elem_content, context_title=" > ".join(breadcrumbs)
                    )
                    tbl_prefix = ContextPrefixer.build_prefix(
                        doc_title=doc.title,
                        doc_type=DocumentType.SEC_FILING,
                        breadcrumbs=breadcrumbs,
                        chunk_type=ChunkType.TABLE
                    )
                    table_chunk = Chunk(
                        doc_id=doc.doc_id,
                        text=formatted_tbl_text,
                        context_prefix=tbl_prefix,
                        metadata={"table_caption": elem_content.caption, **doc.metadata},
                        parent_id=parent_chunk.id,
                        level=2,
                        chunk_type=ChunkType.TABLE,
                        token_count=HierarchyTreeBuilder.estimate_token_count(formatted_tbl_text),
                        breadcrumbs=breadcrumbs,
                        footnotes=footnotes
                    )
                    parent_chunk.child_ids.append(table_chunk.id)
                    chunks.append(table_chunk)
                else:
                    temp_parent = Chunk(
                        doc_id=doc.doc_id,
                        text=elem_content,
                        context_prefix=prefix,
                        metadata=doc.metadata,
                        id=parent_chunk.id,
                        level=1,
                        breadcrumbs=breadcrumbs,
                        footnotes=footnotes
                    )
                    leaf_chunks = HierarchyTreeBuilder.split_into_leaf_chunks(
                        parent_chunk=temp_parent,
                        doc_title=doc.title,
                        doc_type=DocumentType.SEC_FILING
                    )
                    for leaf in leaf_chunks:
                        leaf.text = FootnoteHandler.attach_footnotes_to_chunk_text(leaf.text, footnotes)
                        chunks.append(leaf)

        return chunks

    def _split_by_sec_headings(self, text: str) -> List[Tuple[str, str]]:
        """Splits text on SEC Part and Item headings while preserving heading names."""
        matches = list(SEC_HEADER_REGEX.finditer(text))
        if not matches:
            return [("Overview", text)]

        blocks = []
        # Any text before the first heading
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                blocks.append(("Overview", preamble))

        for i in range(len(matches)):
            heading = matches[i].group(1).strip()
            start_idx = matches[i].end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            body = text[start_idx:end_idx].strip()
            blocks.append((heading, body))

        return blocks
