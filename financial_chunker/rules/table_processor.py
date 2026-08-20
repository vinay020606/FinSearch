"""
Table Processor Rule module.
Identifies, preserves, formats, and extracts narrative representations for financial tables.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from ..types import TableData, Footnote
from .footnote_handler import FootnoteHandler


# Regex patterns for detecting Markdown tables
MARKDOWN_TABLE_REGEX = re.compile(
    r'(?:^\|.*\|\s*\n)(?:^\|[-:\s|]+\|\s*\n)(?:^\|.*\|\s*\n?)+',
    re.MULTILINE
)

# Regex pattern for HTML tables
HTML_TABLE_REGEX = re.compile(
    r'<table[^>]*>.*?</table>',
    re.DOTALL | re.IGNORECASE
)


class TableProcessor:
    """Handles financial table isolation, markdown transformation, and row narrative generation."""

    @staticmethod
    def extract_markdown_tables(text: str) -> List[Tuple[str, TableData]]:
        """
        Extracts markdown formatted tables from raw text.
        Returns tuples of (raw_table_text, TableData).
        """
        extracted = []
        matches = list(MARKDOWN_TABLE_REGEX.finditer(text))

        for match in matches:
            raw_table = match.group(0)
            lines = [line.strip() for line in raw_table.strip().split('\n') if line.strip()]
            if len(lines) < 2:
                continue

            # Parse headers
            header_line = lines[0]
            headers = [col.strip() for col in header_line.strip('|').split('|')]

            # Skip delimiter line (lines[1])
            data_rows = []
            for line in lines[2:]:
                cols = [col.strip() for col in line.strip('|').split('|')]
                if len(cols) == len(headers):
                    data_rows.append(cols)

            table_data = TableData(
                caption="Financial Table",
                headers=headers,
                rows=data_rows,
                raw_markdown=raw_table
            )
            table_data.narrative_summary = TableProcessor.generate_row_narrative(table_data)

            extracted.append((raw_table, table_data))

        return extracted

    @staticmethod
    def generate_row_narrative(table: TableData, title_context: str = "") -> str:
        """
        Converts tabular financial data into dense, clear natural language sentences.
        This provides high semantic match quality for embedding models.
        """
        narratives = []
        ctx = f"Table: {table.caption}" if table.caption else "Financial Data Table"
        if title_context:
            ctx = f"{title_context} - {ctx}"

        headers = table.headers
        if not headers or len(headers) < 2:
            return table.raw_markdown

        row_header_label = headers[0]
        col_headers = headers[1:]

        for row in table.rows:
            if not row or len(row) == 0:
                continue
            item_name = row[0]
            row_values = row[1:]

            row_parts = []
            for col_name, val in zip(col_headers, row_values):
                if val and val != "-" and val != "N/A":
                    row_parts.append(f"{col_name}: {val}")

            if row_parts:
                narrative_line = f"In {ctx}, for '{item_name}', " + ", ".join(row_parts) + "."
                narratives.append(narrative_line)

        return "\n".join(narratives)

    @staticmethod
    def format_table_chunk_text(table: TableData, context_title: str = "") -> str:
        """
        Creates a dual-representation chunk text containing both
        the structured Markdown table and the flattened row narratives.
        """
        parts = []

        if context_title:
            parts.append(f"### Table Context: {context_title}")

        if table.caption and table.caption != "Financial Table":
            parts.append(f"**Caption**: {table.caption}")

        # 1. Standard structured Markdown table
        if table.raw_markdown:
            parts.append(table.raw_markdown)
        else:
            # Reconstruct markdown table
            header_str = "| " + " | ".join(table.headers) + " |"
            delim_str = "| " + " | ".join(["---"] * len(table.headers)) + " |"
            row_strs = ["| " + " | ".join(r) + " |" for r in table.rows]
            parts.append("\n".join([header_str, delim_str] + row_strs))

        # 2. Row narrative representation for vector indexing enhancement
        narrative = TableProcessor.generate_row_narrative(table, context_title)
        if narrative:
            parts.append("\n**Row-by-Row Financial Data Summary:**\n" + narrative)

        # 3. Table Notes / Footnotes
        if table.notes:
            parts.append("\n**Table Notes:**\n" + "\n".join(f"- {note}" for note in table.notes))

        return "\n\n".join(parts)
