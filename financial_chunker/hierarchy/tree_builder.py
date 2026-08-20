"""
Hierarchy Tree Builder module.
Constructs parent-child hierarchical trees of chunk nodes for multi-resolution RAG retrieval.
"""

from typing import List, Dict, Any, Optional
import uuid
from ..types import Chunk, ChunkType, DocumentType
from ..prefix.context_prefixer import ContextPrefixer
from ..rules.number_unit_protector import NumberUnitProtector


class HierarchyTreeBuilder:
    """Builds parent-child hierarchical chunk trees and manages bidirectional linkage."""

    @staticmethod
    def estimate_token_count(text: str) -> int:
        """Rough token count estimation based on whitespace tokenization (approx 1.3 words per token)."""
        words = text.split()
        return int(len(words) * 1.3)

    @staticmethod
    def split_into_leaf_chunks(
        parent_chunk: Chunk,
        doc_title: str,
        doc_type: DocumentType,
        target_token_size: int = 350,
        max_token_size: int = 500,
        overlap_tokens: int = 30
    ) -> List[Chunk]:
        """
        Splits a parent section chunk into child leaf chunks respecting semantic boundaries (paragraphs, sentences)
        and applying minimal overlap (0-50 tokens) strictly when required across prose boundaries.
        """
        if parent_chunk.token_count <= max_token_size or parent_chunk.chunk_type == ChunkType.TABLE:
            # Table chunks or small section chunks remain intact as single child chunks
            child = Chunk(
                doc_id=parent_chunk.doc_id,
                text=parent_chunk.text,
                context_prefix=parent_chunk.context_prefix,
                metadata=parent_chunk.metadata.copy(),
                parent_id=parent_chunk.id,
                level=parent_chunk.level + 1,
                chunk_type=parent_chunk.chunk_type,
                token_count=parent_chunk.token_count,
                breadcrumbs=parent_chunk.breadcrumbs.copy(),
                footnotes=parent_chunk.footnotes.copy()
            )
            parent_chunk.child_ids.append(child.id)
            return [child]

        paragraphs = [p.strip() for p in parent_chunk.text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [parent_chunk.text]

        leaf_chunks: List[Chunk] = []
        current_paragraphs: List[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = HierarchyTreeBuilder.estimate_token_count(para)

            if current_tokens + para_tokens > target_token_size and current_paragraphs:
                # Flush current paragraph block to child chunk
                child_text = "\n\n".join(current_paragraphs)

                # Protect number units from boundary splitting
                protected_text = NumberUnitProtector.protect(child_text)

                prefix = ContextPrefixer.build_prefix(
                    doc_title=doc_title,
                    doc_type=doc_type,
                    breadcrumbs=parent_chunk.breadcrumbs,
                    chunk_type=parent_chunk.chunk_type,
                    additional_metadata=parent_chunk.metadata
                )

                child_chunk = Chunk(
                    doc_id=parent_chunk.doc_id,
                    text=protected_text,
                    context_prefix=prefix,
                    metadata=parent_chunk.metadata.copy(),
                    parent_id=parent_chunk.id,
                    level=parent_chunk.level + 1,
                    chunk_type=parent_chunk.chunk_type,
                    token_count=HierarchyTreeBuilder.estimate_token_count(protected_text),
                    breadcrumbs=parent_chunk.breadcrumbs.copy(),
                    footnotes=parent_chunk.footnotes.copy()
                )

                parent_chunk.child_ids.append(child_chunk.id)
                leaf_chunks.append(child_chunk)

                # Minimal overlap: Keep last sentence / small paragraph slice for boundary continuity
                if overlap_tokens > 0 and len(current_paragraphs) > 0:
                    last_para = current_paragraphs[-1]
                    if HierarchyTreeBuilder.estimate_token_count(last_para) <= overlap_tokens:
                        current_paragraphs = [last_para, para]
                        current_tokens = HierarchyTreeBuilder.estimate_token_count(last_para) + para_tokens
                    else:
                        current_paragraphs = [para]
                        current_tokens = para_tokens
                else:
                    current_paragraphs = [para]
                    current_tokens = para_tokens
            else:
                current_paragraphs.append(para)
                current_tokens += para_tokens

        # Flush remaining paragraphs
        if current_paragraphs:
            child_text = "\n\n".join(current_paragraphs)
            protected_text = NumberUnitProtector.protect(child_text)
            prefix = ContextPrefixer.build_prefix(
                doc_title=doc_title,
                doc_type=doc_type,
                breadcrumbs=parent_chunk.breadcrumbs,
                chunk_type=parent_chunk.chunk_type,
                additional_metadata=parent_chunk.metadata
            )
            child_chunk = Chunk(
                doc_id=parent_chunk.doc_id,
                text=protected_text,
                context_prefix=prefix,
                metadata=parent_chunk.metadata.copy(),
                parent_id=parent_chunk.id,
                level=parent_chunk.level + 1,
                chunk_type=parent_chunk.chunk_type,
                token_count=HierarchyTreeBuilder.estimate_token_count(protected_text),
                breadcrumbs=parent_chunk.breadcrumbs.copy(),
                footnotes=parent_chunk.footnotes.copy()
            )
            parent_chunk.child_ids.append(child_chunk.id)
            leaf_chunks.append(child_chunk)

        return leaf_chunks
