"""
Number-Unit Protector Rule module.
Ensures financial numbers, currency symbols, percentages, scale multipliers, and units are never severed across chunk boundaries.
"""

import re
from typing import List, Tuple

# Non-breaking space used to keep numbers and units atomically bound
NBSP = "\u00A0"

# Patterns matching financial numerical expressions
# Example: $50 million, €12.5B, £100,000, 14.5%, 50 bps, $1.25 per share, Q3 2024, FY23, 5.2x
PATTERNS = [
    # Currency symbol attached or space before number: $ 50 million, € 12.5B
    r'([$€£¥₹]\s*\d+(?:,\d{3})*(?:\.\d+)?)',
    # Number followed by financial units/magnitudes: 50 million, 12.5B, 500k, 25 bps, 14%
    r'(\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand|bps|percent|x|shares|per share|bps|[mMbBkK%]))\b',
    # Range expressions: $10 million - $15 million, 10% - 15%
    r'([$€£¥₹]?\d+(?:\.\d+)?\s*(?:[-–]|to)\s*[$€£¥₹]?\d+(?:\.\d+)?\s*(?:million|billion|trillion|bps|%))',
    # Quarter / Fiscal Year tokens: Q1 2024, FY 2023, 3Q23
    r'\b((?:Q[1-4]|FY)\s*\d{2,4})\b',
]

# Regex to find number + unit where space needs non-breaking enforcement
FINANCIAL_UNIT_REGEX = re.compile(
    r'([$€£¥₹]\s*)?(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|trillion|thousand|bps|percent|x|shares|per share|bps|[mMbBkK%])\b',
    re.IGNORECASE
)

CURRENCY_NUMBER_REGEX = re.compile(
    r'([$€£¥₹])\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
    re.IGNORECASE
)

PER_SHARE_REGEX = re.compile(
    r'(\b\d+(?:\.\d+)?)\s*(per share|a share)\b',
    re.IGNORECASE
)

RANGE_REGEX = re.compile(
    r'(\d+(?:\.\d+)?)\s*(to|-|–)\s*(\d+(?:\.\d+)?)\s*(million|billion|trillion|%|bps|percent)\b',
    re.IGNORECASE
)


class NumberUnitProtector:
    """Protects financial numerical expressions from being split across chunk boundaries."""

    @staticmethod
    def protect(text: str) -> str:
        """
        Replaces standard spaces in number-unit financial expressions with non-breaking spaces.
        This prevents naïve whitespace or word-wrap splitters from separating values from their units.
        """
        if not text:
            return text

        # 1. Bind currency symbol to number: "$ 500" -> "$500" or "$\u00A0500"
        protected = CURRENCY_NUMBER_REGEX.sub(rf'\1{NBSP}\2', text)

        # 2. Bind number to unit/scale: "50 million" -> "50\u00A0million"
        protected = FINANCIAL_UNIT_REGEX.sub(rf'\1\2{NBSP}\3', protected)

        # 3. Bind per-share metrics: "$1.25 per share" -> "$1.25\u00A0per\u00A0share"
        protected = PER_SHARE_REGEX.sub(rf'\1{NBSP}per{NBSP}share', protected)

        # 4. Bind ranges: "10 to 15 million" -> "10\u00A0to\u00A015\u00A0million"
        protected = RANGE_REGEX.sub(rf'\1{NBSP}\2{NBSP}\3{NBSP}\4', protected)

        return protected

    @staticmethod
    def unprotect(text: str) -> str:
        """Converts non-breaking spaces back to regular spaces if needed for clean display."""
        return text.replace(NBSP, " ")

    @staticmethod
    def find_atomic_ranges(text: str) -> List[Tuple[int, int]]:
        """
        Returns list of character index intervals (start, end) that must NOT be split.
        """
        atomic_spans: List[Tuple[int, int]] = []
        combined_regex = re.compile(
            r'|'.join(PATTERNS), re.IGNORECASE
        )

        for match in combined_regex.finditer(text):
            atomic_spans.append((match.start(), match.end()))

        return atomic_spans

    @staticmethod
    def is_safe_split_index(text: str, index: int) -> bool:
        """Checks whether a given character index falls inside an atomic number-unit span."""
        spans = NumberUnitProtector.find_atomic_ranges(text)
        for start, end in spans:
            if start < index < end:
                return False
        return True
