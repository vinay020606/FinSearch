"""
Context Prefixer module.
Generates structured contextual header prefixes prepended to chunks before embedding generation.
"""

from typing import List, Dict, Any, Optional
from ..types import DocumentType, ChunkType


class ContextPrefixer:
    """Constructs explicit, high-signal contextual header prefixes for RAG embedding generation."""

    @staticmethod
    def build_prefix(
        doc_title: str,
        doc_type: DocumentType,
        breadcrumbs: List[str],
        chunk_type: ChunkType = ChunkType.PROSE,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Creates a contextual header prefix string.
        Example Output:
        [Document: Apple 2023 10-K | Type: SEC Filing | Section: Item 7 MD&A > Liquidity and Capital Resources | Format: Table]
        """
        prefix_parts = []

        if doc_title:
            prefix_parts.append(f"Document: {doc_title.strip()}")

        if doc_type and doc_type != DocumentType.AUTO:
            readable_type = doc_type.value.replace('_', ' ').title()
            prefix_parts.append(f"Type: {readable_type}")

        if breadcrumbs:
            # Clean and filter empty breadcrumbs
            clean_bc = [b.strip() for b in breadcrumbs if b and b.strip()]
            if clean_bc:
                path_str = " > ".join(clean_bc)
                prefix_parts.append(f"Section Path: {path_str}")

        if chunk_type and chunk_type != ChunkType.PROSE:
            readable_chunk_type = chunk_type.value.replace('_', ' ').title()
            prefix_parts.append(f"Format: {readable_chunk_type}")

        if additional_metadata:
            speaker = additional_metadata.get("speaker")
            speaker_title = additional_metadata.get("speaker_title")
            if speaker:
                spk_str = f"Speaker: {speaker}"
                if speaker_title:
                    spk_str += f" ({speaker_title})"
                prefix_parts.append(spk_str)

            fiscal_period = additional_metadata.get("fiscal_period")
            if fiscal_period:
                prefix_parts.append(f"Period: {fiscal_period}")

        if not prefix_parts:
            return ""

        return f"[{' | '.join(prefix_parts)}]"
