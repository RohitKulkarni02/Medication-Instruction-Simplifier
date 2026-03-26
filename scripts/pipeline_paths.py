"""Default paths for pipeline artifacts (generated files under ``outputs/``)."""

from __future__ import annotations

import os

OUTPUT_DIR = "outputs"

SIMPLIFIED_LABELS_JSON = os.path.join(OUTPUT_DIR, "simplified_labels.json")
EXTRACTED_ORIGINAL_JSON = os.path.join(OUTPUT_DIR, "extracted_original.json")
EXTRACTED_SIMPLIFIED_JSON = os.path.join(OUTPUT_DIR, "extracted_simplified.json")
EXTRACTED_SIMPLIFIED_FROM_TEXT_JSON = os.path.join(OUTPUT_DIR, "extracted_simplified_from_text.json")
COMPARISON_REPORT_JSON = os.path.join(OUTPUT_DIR, "comparison_report.json")
COMPARISON_REPORT_TEXT_ONLY_JSON = os.path.join(OUTPUT_DIR, "comparison_report_text_only.json")

# Ingest writes here (bare ``drug_labels.json`` also resolves to data/ via ingest helper)
DEFAULT_DRUG_LABELS_JSON = os.path.join("data", "drug_labels.json")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if needed."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
