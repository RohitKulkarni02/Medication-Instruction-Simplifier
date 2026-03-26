from __future__ import annotations

from compare_extractions import compare_pair


def test_dropped_field() -> None:
    r = compare_pair(
        {"drug_name": "X", "dosage": "10 mg", "warnings": "a", "boxed_warning": None, "contraindications": None, "interactions": None},
        {"drug_name": "X", "dosage": "", "warnings": None, "boxed_warning": None, "contraindications": None, "interactions": None},
    )
    dropped = [i for i in r["issues"] if i.get("error_type") == "dropped"]
    assert dropped
    assert dropped[0].get("simplified_value") is None
    assert "original_snippet" in dropped[0]


def test_dose_mismatch() -> None:
    r = compare_pair(
        {"drug_name": "X", "dosage": "Take 10 mg twice daily", "warnings": None, "boxed_warning": None, "contraindications": None, "interactions": None},
        {"drug_name": "X", "dosage": "Take once daily", "warnings": None, "boxed_warning": None, "contraindications": None, "interactions": None},
    )
    types = {i["type"] for i in r["issues"]}
    assert "POSSIBLE_DOSE_MISMATCH" in types
