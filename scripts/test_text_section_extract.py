from __future__ import annotations

from text_section_extract import extract_sections_from_simplified_text


def test_parse_headings() -> None:
    text = """Patient-friendly medication instructions for X

Boxed warning

BW line one

Dosage

Take 1 tablet daily.

Warnings

Do not drive.

Contraindications

None known.

Drug interactions

Avoid alcohol.
"""
    out = extract_sections_from_simplified_text(text)
    assert out["boxed_warning"] == "BW line one"
    assert "Take 1 tablet" in (out["dosage"] or "")
    assert "Do not drive" in (out["warnings"] or "")
    assert "None known" in (out["contraindications"] or "")
    assert "Avoid alcohol" in (out["interactions"] or "")
