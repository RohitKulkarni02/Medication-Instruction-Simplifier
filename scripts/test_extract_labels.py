from __future__ import annotations

from unittest.mock import patch

import pytest

from extract_labels import (
    ORIGINAL_FIELD_MAP,
    SIMPLIFIED_FIELD_MAP,
    extract_original,
    extract_record,
    extract_simplified,
    print_summary,
)


@pytest.fixture
def full_original() -> dict:
    return {
        "drug_name": "ALPRAZOLAM",
        "boxed_warning": "Risk of dependence.",
        "dosage_and_administration": "Take 0.25mg orally.",
        "warnings": "May cause drowsiness.",
        "contraindications": "Do not use with ketoconazole.",
        "drug_interactions": "Avoid CNS depressants.",
    }


@pytest.fixture
def full_simplified() -> dict:
    return {
        "drug_name": "ALPRAZOLAM",
        "boxed_warning": "Risk of dependence.",
        "dosage": "Take 0.25mg orally.",
        "warnings": "May cause drowsiness.",
        "contraindications": "Do not use with ketoconazole.",
        "interactions": "Avoid CNS depressants.",
    }


def _capture_print_summary_output(results: list[dict]) -> str:
    with patch("builtins.print") as mock_print:
        print_summary(results)
        return "\n".join(" ".join(str(a) for a in call.args) for call in mock_print.call_args_list)


def test_all_fields_present_original(full_original: dict) -> None:
    result = extract_record(full_original, ORIGINAL_FIELD_MAP, "openfda")
    assert result["boxed_warning"] == "Risk of dependence."
    assert result["dosage"] == "Take 0.25mg orally."
    assert result["warnings"] == "May cause drowsiness."
    assert result["contraindications"] == "Do not use with ketoconazole."
    assert result["interactions"] == "Avoid CNS depressants."


def test_all_fields_present_simplified(full_simplified: dict) -> None:
    result = extract_record(full_simplified, SIMPLIFIED_FIELD_MAP, "simplified")
    assert result["boxed_warning"] == "Risk of dependence."
    assert result["dosage"] == "Take 0.25mg orally."
    assert result["warnings"] == "May cause drowsiness."
    assert result["contraindications"] == "Do not use with ketoconazole."
    assert result["interactions"] == "Avoid CNS depressants."


def test_missing_field_is_null(full_original: dict) -> None:
    record = dict(full_original)
    del record["drug_interactions"]
    result = extract_record(record, ORIGINAL_FIELD_MAP, "openfda")
    assert result["interactions"] is None


def test_empty_string_is_null(full_original: dict) -> None:
    record = dict(full_original)
    record["warnings"] = ""
    result = extract_record(record, ORIGINAL_FIELD_MAP, "openfda")
    assert result["warnings"] is None


def test_extract_original_length() -> None:
    batch = [
        {
            "drug_name": "A",
            "boxed_warning": "w",
            "dosage_and_administration": "d",
            "warnings": "w",
            "contraindications": "c",
            "drug_interactions": "i",
        },
        {
            "drug_name": "B",
            "boxed_warning": "w",
            "dosage_and_administration": "d",
            "warnings": "w",
            "contraindications": "c",
            "drug_interactions": "i",
        },
    ]
    results = extract_original(batch)
    assert len(results) == 2
    assert all(r["source"] == "openfda" for r in results)


def test_extract_simplified_length() -> None:
    batch = [
        {
            "drug_name": "A",
            "boxed_warning": "w",
            "dosage": "d",
            "warnings": "w",
            "contraindications": "c",
            "interactions": "i",
        },
        {
            "drug_name": "B",
            "boxed_warning": "w",
            "dosage": "d",
            "warnings": "w",
            "contraindications": "c",
            "interactions": "i",
        },
    ]
    results = extract_simplified(batch)
    assert len(results) == 2
    assert all(r["source"] == "simplified" for r in results)


def test_print_summary_counts_nulls() -> None:
    results = [
        {
            "drug_name": "A",
            "boxed_warning": "w",
            "dosage": "d",
            "warnings": None,
            "contraindications": "c",
            "interactions": "i",
        }
    ]
    output = _capture_print_summary_output(results)
    assert "warnings" in output
    assert "1" in output

