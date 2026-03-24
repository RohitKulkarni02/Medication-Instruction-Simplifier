"""
Rule-based extraction of safety sections from simplified label `simplified_text`.

Expects headings produced by `simplify_labels.py`:
  Boxed warning, Dosage, Warnings, Contraindications, Drug interactions
"""

from __future__ import annotations

import re
from typing import Any


# Order matters for parsing; first match wins as section start.
HEADING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("boxed_warning", re.compile(r"^boxed warning\s*$", re.I)),
    ("dosage", re.compile(r"^dosage\s*$", re.I)),
    ("warnings", re.compile(r"^warnings\s*$", re.I)),
    ("contraindications", re.compile(r"^contraindications\s*$", re.I)),
    ("interactions", re.compile(r"^drug interactions\s*$", re.I)),
]


def _which_heading(line: str) -> str | None:
    stripped = line.strip()
    for field, pat in HEADING_PATTERNS:
        if pat.match(stripped):
            return field
    return None


def extract_sections_from_simplified_text(simplified_text: str | None) -> dict[str, str | None]:
    """
    Parse free-form `simplified_text` into the same field names as structured extraction.
    Returns nulls for sections not found or empty.
    """
    if not simplified_text or not str(simplified_text).strip():
        return {
            "boxed_warning": None,
            "dosage": None,
            "warnings": None,
            "contraindications": None,
            "interactions": None,
        }

    lines = simplified_text.replace("\r\n", "\n").split("\n")
    current: str | None = None
    buffers: dict[str, list[str]] = {k: [] for k, _ in HEADING_PATTERNS}

    for raw in lines:
        line = raw.rstrip()
        heading = _which_heading(line)
        if heading is not None:
            current = heading
            continue
        if current:
            buffers[current].append(line)

    result: dict[str, str | None] = {}
    for field, _ in HEADING_PATTERNS:
        text = "\n".join(buffers[field]).strip()
        result[field] = text if text else None

    return result


def merge_structured_and_text(
    record: dict[str, Any],
    *,
    prefer_text: bool = False,
) -> dict[str, Any]:
    """
    Combine structured keys on `record` with sections parsed from `simplified_text`.
    If prefer_text is True, non-null parsed sections override structured fields.
    """
    drug_name = record.get("drug_name", "UNKNOWN")
    from_text = extract_sections_from_simplified_text(record.get("simplified_text"))

    merged: dict[str, Any] = {
        "drug_name": drug_name,
        "source": "simplified",
        "extraction_method": "rule_based_text",
        "extraction_confidence": "medium",
    }

    struct_map = {
        "boxed_warning": record.get("boxed_warning"),
        "dosage": record.get("dosage"),
        "warnings": record.get("warnings"),
        "contraindications": record.get("contraindications"),
        "interactions": record.get("interactions"),
    }

    for field in ["boxed_warning", "dosage", "warnings", "contraindications", "interactions"]:
        tv = from_text.get(field)
        sv_raw = struct_map.get(field)
        sv = str(sv_raw).strip() if sv_raw else None
        if prefer_text:
            val = tv or sv
        else:
            val = sv or tv
        merged[field] = val if val else None

    return merged
