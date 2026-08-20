"""
Unit tests for TableProcessor.
"""

from financial_chunker.rules.table_processor import TableProcessor


def test_markdown_table_narrative_extraction():
    raw_md_table = """| Metric | Q3 2023 | Q3 2022 |
| --- | --- | --- |
| Revenue | $89.5B | $90.1B |
| Net Income | $23.0B | $20.7B |"""

    tables = TableProcessor.extract_markdown_tables(raw_md_table)
    assert len(tables) == 1

    _, table_data = tables[0]
    narrative = table_data.narrative_summary
    assert "Revenue" in narrative
    assert "Q3 2023: $89.5B" in narrative

    formatted = TableProcessor.format_table_chunk_text(table_data, context_title="Income Statement")
    assert "### Table Context: Income Statement" in formatted
    assert "Row-by-Row Financial Data Summary" in formatted


test_markdown_table_narrative_extraction()
print("TableProcessor tests passed!")
