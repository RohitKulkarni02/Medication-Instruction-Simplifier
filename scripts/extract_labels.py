"""
Layer 1 Structured Extractor.

Extracts safety-critical fields from two supported data sources into one unified schema:
  - original    -> openFDA / ingest output (raw labels)
  - simplified  -> simplified label output (LLM or local)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

from pipeline_paths import (
    DEFAULT_DRUG_LABELS_JSON,
    EXTRACTED_ORIGINAL_JSON,
    EXTRACTED_SIMPLIFIED_JSON,
    SIMPLIFIED_LABELS_JSON,
    ensure_parent_dir,
)
from text_section_extract import extract_sections_from_simplified_text, merge_structured_and_text

SAFETY_FIELDS = ["boxed_warning", "dosage", "warnings", "contraindications", "interactions"]

ORIGINAL_FIELD_MAP = {
    "boxed_warning": "boxed_warning",
    "dosage": "dosage_and_administration",
    "warnings": "warnings",
    "contraindications": "contraindications",
    "interactions": "drug_interactions",
}

SIMPLIFIED_FIELD_MAP = {
    "boxed_warning": "boxed_warning",
    "dosage": "dosage",
    "warnings": "warnings",
    "contraindications": "contraindications",
    "interactions": "interactions",
}


def extract_record(record: dict, field_map: dict[str, str], source: str) -> dict:
    """Map input keys to unified safety fields for one label record.

    Parameters:
        record: Raw label dict; keys depend on ``field_map`` (openfda vs simplified shape).
        field_map: Maps each unified output field name to the source key in ``record``.
        source: Stored as ``source`` on the result (e.g. "openfda" or "simplified").

    Returns:
        Dict in the unified extraction schema (metadata plus five safety fields).

    Side effects:
        Prints a WARNING when a mapped field is missing, empty, or whitespace-only string.
    """
    drug_name = record.get("drug_name", "UNKNOWN")
    extracted = {
        "drug_name": drug_name,
        "source": source,
        "extraction_method": "structured",
        "extraction_confidence": "high",
    }

    for output_field, input_key in field_map.items():
        value = record.get(input_key)
        if value is None:
            print(f"WARNING: {drug_name} missing field: {output_field}")
            extracted[output_field] = None
            continue

        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                print(f"WARNING: {drug_name} missing field: {output_field}")
                extracted[output_field] = None
            else:
                extracted[output_field] = cleaned
        else:
            extracted[output_field] = value

    return extracted


def extract_original(data: list[dict]) -> list[dict]:
    """Extract unified records from openfda-shaped label dicts.

    Parameters:
        data: List of raw records using keys from ingest (e.g. ``dosage_and_administration``).

    Returns:
        List of extracted dicts with ``source`` set to "openfda".
    """
    return [extract_record(record, ORIGINAL_FIELD_MAP, "openfda") for record in data]


def extract_simplified(data: list[dict]) -> list[dict]:
    """Extract unified records from simplified structured JSON label dicts.

    Parameters:
        data: List of raw records using simplified keys (e.g. ``dosage``, ``interactions``).

    Returns:
        List of extracted dicts with ``source`` set to "simplified".
    """
    return [extract_record(record, SIMPLIFIED_FIELD_MAP, "simplified") for record in data]


def extract_simplified_from_text_only(data: list[dict]) -> list[dict]:
    """Build unified records by parsing ``simplified_text`` only (no structured JSON fields).

    Parameters:
        data: List of dicts expected to contain ``drug_name`` and ``simplified_text``.

    Returns:
        List of extracted dicts with ``extraction_method`` ``rule_based_text`` and
        ``extraction_confidence`` ``medium``.

    Side effects:
        Prints a WARNING for each safety field missing after text parsing.
    """
    out: list[dict] = []
    for record in data:
        drug_name = record.get("drug_name", "UNKNOWN")
        sections = extract_sections_from_simplified_text(record.get("simplified_text"))
        row: dict = {
            "drug_name": drug_name,
            "source": "simplified",
            "extraction_method": "rule_based_text",
            "extraction_confidence": "medium",
        }
        for field in SAFETY_FIELDS:
            v = sections.get(field)
            if not v:
                print(f"WARNING: {drug_name} missing field (from text): {field}")
                row[field] = None
            else:
                row[field] = v
        out.append(row)
    return out


def extract_simplified_hybrid(data: list[dict]) -> list[dict]:
    """Merge structured simplified fields with sections parsed from ``simplified_text``.

    Parameters:
        data: List of simplified records that may include both JSON fields and ``simplified_text``.

    Returns:
        List of unified extracted dicts; gaps in structured data are filled from parsed text
        (``prefer_text=False`` keeps structured values when present).
    """
    return [merge_structured_and_text(record, prefer_text=False) for record in data]


def print_summary(results: list[dict]) -> None:
    """Print counts of null safety fields across extracted records.

    Parameters:
        results: List of dicts in the unified extraction schema.

    Returns:
        None. Writes summary lines to stdout.
    """
    total = len(results)
    null_field_counts: Counter[str] = Counter()
    records_with_nulls = 0

    for record in results:
        has_null = False
        for field in SAFETY_FIELDS:
            if record.get(field) is None:
                null_field_counts[field] += 1
                has_null = True
        if has_null:
            records_with_nulls += 1

    print("\n--- Extraction Summary ---")
    print(f"Total drugs processed:        {total}")
    print(f"Records with >=1 null field:  {records_with_nulls}")

    if null_field_counts:
        print("Null counts by field (most common first):")
        for field, count in null_field_counts.most_common():
            print(f"  {field:<20} {count}")
    else:
        print("All fields present for all records.")


def main() -> int:
    """CLI: load JSON labels, run the selected extractor, write output JSON and summary.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Layer 1 structured extractor for medication safety fields."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["original", "simplified"],
        help="Input source: 'original' (openfda) or 'simplified' (simplified label JSON).",
    )
    parser.add_argument("--input", type=str, default=None, help="Optional input JSON path override.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path override.")
    parser.add_argument(
        "--simplified-mode",
        type=str,
        default="structured",
        choices=["structured", "from_text", "hybrid"],
        help="For --source simplified: structured JSON keys only; from_text parses simplified_text; "
        "hybrid fills gaps from simplified_text.",
    )
    args = parser.parse_args()

    if args.source == "original":
        input_file = args.input or DEFAULT_DRUG_LABELS_JSON
        output_file = args.output or EXTRACTED_ORIGINAL_JSON
        extractor = extract_original
    else:
        input_file = args.input or SIMPLIFIED_LABELS_JSON
        output_file = args.output or EXTRACTED_SIMPLIFIED_JSON
        mode = getattr(args, "simplified_mode", "structured")
        if mode == "from_text":
            extractor = extract_simplified_from_text_only
        elif mode == "hybrid":
            extractor = extract_simplified_hybrid
        else:
            extractor = extract_simplified

    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Extracting {len(data)} records (source: {args.source})...")
    results = extractor(data)

    ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Output written to {output_file}")
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

