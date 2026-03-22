"""
test_extract_labels.py — Pytest tests for extract_labels.py

Tests:
  - extract_record()
  - extract_original()
  - extract_simplified()
  - print_summary()

Run:
    pytest test_extract_labels.py -v
"""

import pytest
from unittest.mock import patch

from extract_labels import (
    extract_record,
    extract_original,
    extract_simplified,
    print_summary,
    ORIGINAL_FIELD_MAP,
    SIMPLIFIED_FIELD_MAP,
)


@pytest.fixture
def full_original():
    return {
        "drug_name": "ALPRAZOLAM",
        "boxed_warning": "Risk of dependence.",
        "dosage_and_administration": "Take 0.25mg orally.",
        "warnings": "May cause drowsiness.",
        "contraindications": "Do not use with ketoconazole.",
        "drug_interactions": "Avoid CNS depressants.",
    }


@pytest.fixture
def full_simplified():
    return {
        "drug_name": "ALPRAZOLAM",
        "boxed_warning": "Risk of dependence.",
        "dosage": "Take 0.25mg orally.",
        "warnings": "May cause drowsiness.",
        "contraindications": "Do not use with ketoconazole.",
        "interactions": "Avoid CNS depressants.",
    }


@pytest.fixture
def original_batch():
    base = {
        "boxed_warning": "Warning text.",
        "dosage_and_administration": "Dosage text.",
        "warnings": "Warnings text.",
        "contraindications": "Contraindications text.",
        "drug_interactions": "Interactions text.",
    }
    return [
        dict(base, drug_name="ALPRAZOLAM"),
        dict(base, drug_name="LISINOPRIL"),
    ]


@pytest.fixture
def simplified_batch():
    base = {
        "boxed_warning": "Warning text.",
        "dosage": "Dosage text.",
        "warnings": "Warnings text.",
        "contraindications": "Contraindications text.",
        "interactions": "Interactions text.",
    }
    return [
        dict(base, drug_name="ALPRAZOLAM"),
        dict(base, drug_name="LISINOPRIL"),
    ]


def _capture_print_summary_output(results):
    with patch("builtins.print") as mock_print:
        print_summary(results)
        lines = []
        for call in mock_print.call_args_list:
            lines.append(" ".join(str(a) for a in call.args))
        return "\n".join(lines)


class TestExtractRecord:
    """extract_record() — shared field-mapping core."""

    def test_all_fields_present_original(self, full_original):
        result = extract_record(full_original, ORIGINAL_FIELD_MAP, "openfda")
        assert result["boxed_warning"] == "Risk of dependence."
        assert result["dosage"] == "Take 0.25mg orally."
        assert result["warnings"] == "May cause drowsiness."
        assert result["contraindications"] == "Do not use with ketoconazole."
        assert result["interactions"] == "Avoid CNS depressants."

    def test_metadata_fields_original(self, full_original):
        result = extract_record(full_original, ORIGINAL_FIELD_MAP, "openfda")
        assert result["extraction_method"] == "structured"
        assert result["extraction_confidence"] == "high"
        assert result["source"] == "openfda"
        assert result["drug_name"] == "ALPRAZOLAM"

    def test_missing_field_is_null(self, full_original):
        record = dict(full_original)
        del record["drug_interactions"]
        result = extract_record(record, ORIGINAL_FIELD_MAP, "openfda")
        assert result["interactions"] is None

    def test_empty_string_is_null(self, full_original):
        record = dict(full_original)
        record["warnings"] = ""
        result = extract_record(record, ORIGINAL_FIELD_MAP, "openfda")
        assert result["warnings"] is None

    def test_whitespace_only_is_null(self, full_original):
        record = dict(full_original)
        record["warnings"] = "   "
        result = extract_record(record, ORIGINAL_FIELD_MAP, "openfda")
        assert result["warnings"] is None

    def test_missing_drug_name_defaults_to_unknown(self, full_original):
        record = dict(full_original)
        del record["drug_name"]
        result = extract_record(record, ORIGINAL_FIELD_MAP, "openfda")
        assert result["drug_name"] == "UNKNOWN"

    def test_all_fields_present_simplified(self, full_simplified):
        result = extract_record(full_simplified, SIMPLIFIED_FIELD_MAP, "simplified")
        assert result["boxed_warning"] == "Risk of dependence."
        assert result["dosage"] == "Take 0.25mg orally."
        assert result["warnings"] == "May cause drowsiness."
        assert result["contraindications"] == "Do not use with ketoconazole."
        assert result["interactions"] == "Avoid CNS depressants."

    def test_simplified_uses_interactions_key(self):
        record = {
            "drug_name": "LISINOPRIL",
            "boxed_warning": "Fetal toxicity.",
            "dosage": "10mg once daily.",
            "warnings": "Monitor renal function.",
            "contraindications": "Avoid in pregnancy.",
            "interactions": "NSAIDs reduce effect.",
        }
        result = extract_record(record, SIMPLIFIED_FIELD_MAP, "simplified")
        assert result["interactions"] == "NSAIDs reduce effect."
        assert result["interactions"] is not None


class TestExtractOriginal:
    """extract_original() — openfda drug label records."""

    def test_returns_list_of_same_length(self, original_batch):
        results = extract_original(original_batch)
        assert len(results) == 2

    def test_source_is_openfda(self, original_batch):
        results = extract_original(original_batch)
        for record in results:
            assert record["source"] == "openfda"


class TestExtractSimplified:
    """extract_simplified() — simplified drug label records."""

    def test_returns_list_of_same_length(self, simplified_batch):
        results = extract_simplified(simplified_batch)
        assert len(results) == 2

    def test_source_is_simplified(self, simplified_batch):
        results = extract_simplified(simplified_batch)
        for record in results:
            assert record["source"] == "simplified"


class TestPrintSummary:
    """print_summary() — extraction statistics to stdout."""

    def test_all_fields_present_prints_no_nulls(self):
        results = [
            {
                "drug_name": "ALPRAZOLAM",
                "boxed_warning": "Text.",
                "dosage": "Text.",
                "warnings": "Text.",
                "contraindications": "Text.",
                "interactions": "Text.",
            }
        ]
        output = _capture_print_summary_output(results)
        assert "All fields present" in output

    def test_null_fields_counted_correctly(self):
        results = [
            {
                "drug_name": "ALPRAZOLAM",
                "boxed_warning": "Text.",
                "dosage": "Text.",
                "warnings": None,
                "contraindications": "Text.",
                "interactions": "Text.",
            }
        ]
        output = _capture_print_summary_output(results)
        assert "warnings" in output
        assert "1" in output

    def test_records_with_nulls_count(self):
        results = [
            {
                "drug_name": "ALPRAZOLAM",
                "boxed_warning": "Text.",
                "dosage": "Text.",
                "warnings": None,
                "contraindications": "Text.",
                "interactions": "Text.",
            },
            {
                "drug_name": "LISINOPRIL",
                "boxed_warning": "Text.",
                "dosage": "Text.",
                "warnings": "Text.",
                "contraindications": "Text.",
                "interactions": "Text.",
            },
        ]
        output = _capture_print_summary_output(results)
        assert "1" in output
        assert "2" in output
