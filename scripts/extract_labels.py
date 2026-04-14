"""
Layer 1 Structured Extractor.

Extracts safety-critical fields from two supported data sources into one unified schema:
  - original    -> openFDA / ingest output (raw labels)
  - simplified  -> simplified label output (LLM or local)
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

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
    return [extract_record(record, ORIGINAL_FIELD_MAP, "openfda") for record in data]


def extract_simplified(data: list[dict]) -> list[dict]:
    return [extract_record(record, SIMPLIFIED_FIELD_MAP, "simplified") for record in data]


def extract_simplified_from_text_only(data: list[dict]) -> list[dict]:
    """Parse only `simplified_text` (no structured JSON fields)."""
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
    """Structured fields first; fill gaps from `simplified_text` headings."""
    return [merge_structured_and_text(record, prefer_text=False) for record in data]


def print_summary(results: list[dict]) -> None:
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
        input_file = args.input or "data/drug_labels.json"
        output_file = args.output or "outputs/extracted_original.json"
        extractor = extract_original
    else:
        input_file = args.input or "outputs/simplified_labels.json"
        output_file = args.output or "outputs/extracted_simplified.json"
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

    out_dir = os.path.dirname(os.path.abspath(output_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Output written to {output_file}")
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

