"""
Unit tests for NumberUnitProtector.
"""

from financial_chunker.rules.number_unit_protector import NumberUnitProtector, NBSP


def test_protect_currency_and_magnitudes():
    raw = "Revenue increased by 50 million to 1.25 billion in Q3 2024."
    protected = NumberUnitProtector.protect(raw)
    assert f"50{NBSP}million" in protected
    assert f"1.25{NBSP}billion" in protected


def test_protect_percentages_and_bps():
    raw = "Operating margin expanded by 150 bps to 22.5% YoY."
    protected = NumberUnitProtector.protect(raw)
    assert f"150{NBSP}bps" in protected
    assert f"22.5{NBSP}%" in protected or "22.5%" in protected


test_protect_currency_and_magnitudes()
test_protect_percentages_and_bps()
print("NumberUnitProtector tests passed!")
