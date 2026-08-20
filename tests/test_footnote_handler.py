"""
Unit tests for FootnoteHandler.
"""

from financial_chunker.rules.footnote_handler import FootnoteHandler
from financial_chunker.types import Footnote


def test_extract_and_attach_footnotes():
    raw_text = """Net Income (1) reached $4.2B for FY23, up 12% from prior period.

(1) Net income includes non-recurring acquisition costs of $15M."""

    clean_text, footnotes = FootnoteHandler.extract_footnote_definitions(raw_text)
    assert len(footnotes) == 1
    assert footnotes[0].marker == "1"
    assert "non-recurring acquisition costs" in footnotes[0].text

    attached_text = FootnoteHandler.attach_footnotes_to_chunk_text(clean_text, footnotes)
    assert "--- Footnotes ---" in attached_text
    assert "[1]" in attached_text


test_extract_and_attach_footnotes()
print("FootnoteHandler tests passed!")
