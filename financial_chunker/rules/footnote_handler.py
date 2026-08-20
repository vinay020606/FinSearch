"""
Footnote Handler Rule module.
Detects footnote references, extracts footnote definitions, and re-attaches footnotes to relevant chunks.
"""

import re
from typing import List, Dict, Tuple, Optional
from ..types import Footnote


# Regular expressions for footnote reference markers in body prose
# e.g., (1), [2], *, †, ‡, [a], (a)
REF_MARKER_REGEX = re.compile(
    r'(?:\((\d+|[a-z]|\*|†|‡)\)|\[(\d+|[a-z]|\*|†|‡)\]|\*|†|‡)',
    re.IGNORECASE
)

# Regular expressions for footnote definitions at section/table bottoms
# e.g., "(1) Net income includes $5M non-recurring charge." or "[a] Adjusted EBITDA..." or "* Represents GAAP measure."
DEFINITION_REGEX = re.compile(
    r'^\s*(?:\((\d+|[a-z]|\*|†|‡)\)|\[(\d+|[a-z]|\*|†|‡)\]|\*|†|‡|Note\s*(\d+|[a-z])):?\s+(.+)$',
    re.MULTILINE | re.IGNORECASE
)


class FootnoteHandler:
    """Detects, extracts, and attaches footnotes to document chunks."""

    @staticmethod
    def extract_footnote_definitions(text: str) -> Tuple[str, List[Footnote]]:
        """
        Parses text for trailing footnote definitions, extracts them, and returns
        the cleaned text alongside extracted Footnote objects.
        """
        footnotes: List[Footnote] = []
        lines = text.split('\n')
        clean_lines: List[str] = []

        for line in lines:
            match = DEFINITION_REGEX.match(line)
            if match:
                marker = (match.group(1) or match.group(2) or match.group(3) or "*").strip()
                fn_text = match.group(4).strip()
                footnotes.append(Footnote(marker=marker, text=fn_text))
            else:
                clean_lines.append(line)

        clean_text = '\n'.join(clean_lines)
        return clean_text, footnotes

    @staticmethod
    def find_referenced_markers(text: str) -> List[str]:
        """Finds all footnote reference markers embedded within the chunk text."""
        markers = set()
        for match in REF_MARKER_REGEX.finditer(text):
            marker = match.group(1) or match.group(2) or match.group(0)
            if marker:
                markers.add(marker.strip('()[]'))
        return list(markers)

    @staticmethod
    def attach_footnotes_to_chunk_text(chunk_text: str, footnotes: List[Footnote]) -> str:
        """
        Appends relevant footnote definitions to the bottom of the chunk text
        so that vector embeddings and LLM retrievals have immediate access to footnote definitions.
        """
        if not footnotes:
            return chunk_text

        referenced_markers = FootnoteHandler.find_referenced_markers(chunk_text)
        relevant_fn = [fn for fn in footnotes if fn.marker in referenced_markers or fn.marker == "*"]

        # If no specific marker match, but footnotes exist for the block, attach all
        fn_to_attach = relevant_fn if relevant_fn else footnotes

        if not fn_to_attach:
            return chunk_text

        fn_block_lines = ["\n\n--- Footnotes ---"]
        for fn in fn_to_attach:
            fn_block_lines.append(f"[{fn.marker}] {fn.text}")

        return chunk_text + "\n" + "\n".join(fn_block_lines)
