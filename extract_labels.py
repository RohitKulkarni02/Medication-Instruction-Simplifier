"""
extract_labels.py — Layer 1 Structured Extractor

Extracts safety-critical fields from FDA medication labels into a unified schema.
Supports two input sources:
  - "original"   : drug_labels.json (Chris's openfda data)
  - "simplified" : simplified_labels.json (Rohit's LLM-simplified data)

Usage:
    python extract_labels.py --source original
    python extract_labels.py --source simplified
"""

import json
import argparse
from collections import Counter

# Fields in the unified output schema (excluding metadata fields)
SAFETY_FIELDS = ["boxed_warning", "dosage", "warnings", "contraindications", "interactions"]

# Maps output field name → input key for Chris's openfda data
ORIGINAL_FIELD_MAP = {
    "boxed_warning":     "boxed_warning",
    "dosage":            "dosage_and_administration",
    "warnings":          "warnings",
    "contraindications": "contraindications",
    "interactions":      "drug_interactions",
}

# Maps output field name → input key for Rohit's simplified data
SIMPLIFIED_FIELD_MAP = {
    "boxed_warning":     "boxed_warning",
    "dosage":            "dosage",
    "warnings":          "warnings",
    "contraindications": "contraindications",
    "interactions":      "interactions",
}


def extract_record(record, field_map, source):
    """
    Map a single input record to the unified output schema.

    Iterates over each safety field using the provided field_map. If a field is
    absent or empty in the input, its value is set to None and a WARNING is printed.

    Args:
        record (dict): A single drug record from the input JSON array.
        field_map (dict): Maps output field name → input key name.
        source (str): Either "openfda" or "simplified".

    Returns:
        dict: Extracted record in the unified output schema.
    """
    drug_name = record.get("drug_name", "UNKNOWN")

    extracted = {
        "drug_name":             drug_name,
        "source":                source,
        "extraction_method":     "structured",
        "extraction_confidence": "high",
    }

    for output_field, input_key in field_map.items():
        value = record.get(input_key)
        # Treat missing keys and empty strings both as null
        if not value or not value.strip():
            print(f"WARNING: {drug_name} missing field: {output_field}")
            extracted[output_field] = None
        else:
            extracted[output_field] = value

    return extracted


def extract_original(data):
    """
    Extract safety fields from Chris's openfda drug label records.

    Each record is expected to have top-level keys such as
    dosage_and_administration, warnings, contraindications,
    drug_interactions, and boxed_warning.

    Args:
        data (list[dict]): Parsed JSON array from drug_labels.json.

    Returns:
        list[dict]: List of extracted records in the unified output schema.
    """
    return [extract_record(record, ORIGINAL_FIELD_MAP, "openfda") for record in data]


def extract_simplified(data):
    """
    Extract safety fields from Rohit's simplified drug label records.

    Each record is expected to have top-level keys such as
    dosage, warnings, contraindications, interactions, and boxed_warning.

    Args:
        data (list[dict]): Parsed JSON array from simplified_labels.json.

    Returns:
        list[dict]: List of extracted records in the unified output schema.
    """
    return [extract_record(record, SIMPLIFIED_FIELD_MAP, "simplified") for record in data]


def print_summary(results):
    """
    Print a summary report of the extraction run.

    Reports:
      - Total number of drug records processed
      - Number of records that had at least one null field
      - Per-field null counts, sorted from most to least common

    Args:
        results (list[dict]): List of extracted records.
    """
    total = len(results)
    null_field_counts = Counter()
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
    print(f"Records with ≥1 null field:   {records_with_nulls}")

    if null_field_counts:
        print("Null counts by field (most common first):")
        for field, count in null_field_counts.most_common():
            print(f"  {field:<20} {count}")
    else:
        print("All fields present for all records.")


def main():
    """
    Entry point. Parse --source flag, load input, run extraction, write output.

    --source original   : reads drug_labels.json, writes extracted_original.json
    --source simplified : reads simplified_labels.json, writes extracted_simplified.json
    """
    parser = argparse.ArgumentParser(
        description="Layer 1 structured extractor for medication safety fields."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["original", "simplified"],
        help="Input source: 'original' (openfda) or 'simplified' (Rohit's data)",
    )
    args = parser.parse_args()

    if args.source == "original":
        input_file = "drug_labels.json"
        output_file = "extracted_original.json"
        extractor = extract_original
    else:
        input_file = "simplified_labels.json"
        output_file = "extracted_simplified.json"
        extractor = extract_simplified

    print(f"Loading {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Extracting {len(data)} records (source: {args.source})...")
    results = extractor(data)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Output written to {output_file}")
    print_summary(results)


if __name__ == "__main__":
    main()
