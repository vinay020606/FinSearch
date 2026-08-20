"""
Earnings Call Transcript Chunking Strategy.
Specialized for corporate earnings call transcripts, speaker turn isolation, and Q&A pair preservation.
"""

import re
from typing import List, Tuple, Dict, Any
from .base import BaseChunkingStrategy
from ..types import Document, Chunk, ChunkType, DocumentType
from ..rules.number_unit_protector import NumberUnitProtector
from ..prefix.context_prefixer import ContextPrefixer
from ..hierarchy.tree_builder import HierarchyTreeBuilder

# Regex matching major section headers in transcripts
SECTION_HEADER_REGEX = re.compile(
    r'(?:^|\n)\s*(PREPARED REMARKS|EXECUTIVE PRESENTATION|QUESTION AND ANSWER(?:S)?|Q&A(?:\s+SESSION)?)\s*(?=\n|$)',
    re.IGNORECASE
)

# Regex matching speaker lines: "Tim Cook (CEO):", "Operator:", "Toni Sacconaghi (Sanford Bernstein):"
SPEAKER_REGEX = re.compile(
    r'(?:^|\n)\s*([A-Z][A-Za-z0-9\s\.\,\-\'\(\)]+?):\s+',
    re.MULTILINE
)


class EarningsTranscriptStrategy(BaseChunkingStrategy):
    """Chunking strategy optimized for Earnings Call Transcripts and Q&A sessions."""

    def chunk(self, doc: Document) -> List[Chunk]:
        raw_text = doc.content
        chunks: List[Chunk] = []

        # 1. Document Root
        doc_root = Chunk(
            doc_id=doc.doc_id,
            text=f"Earnings Call Transcript: {doc.title}",
            context_prefix=f"[Document: {doc.title} | Type: Earnings Call]",
            metadata=doc.metadata,
            level=0,
            chunk_type=ChunkType.HEADER,
            breadcrumbs=[doc.title]
        )
        chunks.append(doc_root)

        # 2. Split into Major Sections (Prepared Remarks vs Q&A)
        major_sections = self._split_major_sections(raw_text)

        for sec_name, sec_body in major_sections:
            speaker_turns = self._parse_speaker_turns(sec_body)

            for speaker, text in speaker_turns:
                clean_speaker = speaker.strip()
                clean_text = NumberUnitProtector.protect(text.strip())
                if not clean_text:
                    continue

                breadcrumbs = [doc.title, sec_name]

                speaker_title = ""
                clean_speaker_name = clean_speaker
                if "(" in clean_speaker and ")" in clean_speaker:
                    parts = clean_speaker.split("(")
                    clean_speaker_name = parts[0].strip()
                    speaker_title = parts[1].replace(")", "").strip()

                meta = {
                    "speaker": clean_speaker_name,
                    "speaker_title": speaker_title,
                    "section": sec_name,
                    **doc.metadata
                }

                chunk_kind = ChunkType.QA_PAIR if "Q&A" in sec_name or "QUESTION" in sec_name.upper() else ChunkType.EXECUTIVE_REMARKS

                prefix = ContextPrefixer.build_prefix(
                    doc_title=doc.title,
                    doc_type=DocumentType.EARNINGS_TRANSCRIPT,
                    breadcrumbs=breadcrumbs,
                    chunk_type=chunk_kind,
                    additional_metadata=meta
                )

                turn_text = f"**{clean_speaker}**: {clean_text}"
                turn_chunk = Chunk(
                    doc_id=doc.doc_id,
                    text=turn_text,
                    context_prefix=prefix,
                    metadata=meta,
                    parent_id=doc_root.id,
                    level=1,
                    chunk_type=chunk_kind,
                    token_count=HierarchyTreeBuilder.estimate_token_count(turn_text),
                    breadcrumbs=breadcrumbs
                )
                doc_root.child_ids.append(turn_chunk.id)
                chunks.append(turn_chunk)

                # Leaf chunks
                leaf_chunks = HierarchyTreeBuilder.split_into_leaf_chunks(
                    parent_chunk=turn_chunk,
                    doc_title=doc.title,
                    doc_type=DocumentType.EARNINGS_TRANSCRIPT
                )
                chunks.extend(leaf_chunks)

        return chunks

    def _split_major_sections(self, text: str) -> List[Tuple[str, str]]:
        """Splits transcript text into major sections (Prepared Remarks, Q&A)."""
        matches = list(SECTION_HEADER_REGEX.finditer(text))
        if not matches:
            return [("Prepared Remarks", text)]

        sections = []
        if matches[0].start() > 0:
            preamble = text[:matches[0].start()].strip()
            if preamble:
                sections.append(("Prepared Remarks", preamble))

        for i in range(len(matches)):
            sec_name = matches[i].group(1).strip()
            start_idx = matches[i].end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            body = text[start_idx:end_idx].strip()

            # Normalize section names
            if "QUESTION" in sec_name.upper() or "Q&A" in sec_name.upper():
                norm_name = "Q&A Session"
            else:
                norm_name = "Prepared Remarks"

            sections.append((norm_name, body))

        return sections

    def _parse_speaker_turns(self, text: str) -> List[Tuple[str, str]]:
        """Parses transcript text into list of (speaker_name, dialogue_text)."""
        matches = list(SPEAKER_REGEX.finditer(text))
        if not matches:
            return [("Speaker", text)]

        turns = []
        for i in range(len(matches)):
            speaker = matches[i].group(1).strip()
            start_idx = matches[i].end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            dialogue = text[start_idx:end_idx].strip()
            turns.append((speaker, dialogue))

        return turns
