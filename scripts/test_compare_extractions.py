from __future__ import annotations

from compare_extractions import compare_files, compare_pair


def test_dropped_field() -> None:
    r = compare_pair(
        {"drug_name": "X", "dosage": "10 mg", "warnings": "a", "boxed_warning": None, "contraindications": None, "interactions": None},
        {"drug_name": "X", "dosage": "", "warnings": None, "boxed_warning": None, "contraindications": None, "interactions": None},
    )
    dropped = [i for i in r["issues"] if i.get("error_type") == "dropped"]
    assert dropped
    assert dropped[0].get("simplified_value") is None
    assert "original_snippet" in dropped[0]
    assert dropped[0].get("message")
    assert dropped[0].get("drug_name") == "X"


def test_dose_mismatch() -> None:
    r = compare_pair(
        {"drug_name": "X", "dosage": "Take 10 mg twice daily", "warnings": None, "boxed_warning": None, "contraindications": None, "interactions": None},
        {"drug_name": "X", "dosage": "Take once daily", "warnings": None, "boxed_warning": None, "contraindications": None, "interactions": None},
    )
    dose_issues = [i for i in r["issues"] if i.get("error_type") == "possible_dose_mismatch"]
    assert dose_issues
    assert dose_issues[0].get("missing_tokens")
    assert "original_snippet" in dose_issues[0]
    assert "simplified_snippet" in dose_issues[0]
    assert dose_issues[0].get("message")


def test_compare_files_no_match_uniform_issue(tmp_path) -> None:
    """Original row with no simplified pair emits no_match with field null."""
    orig = tmp_path / "o.json"
    simp = tmp_path / "s.json"
    orig.write_text('[{"drug_name": "ONLYORIG", "boxed_warning": null, "dosage": "d", "warnings": "w", "contraindications": "c", "interactions": "i"}]', encoding="utf-8")
    simp.write_text("[]", encoding="utf-8")
    report = compare_files(str(orig), str(simp))
    assert report.get("schema_version") == "1"
    assert report["per_drug"][0]["issues"][0]["error_type"] == "no_match"
    assert report["per_drug"][0]["issues"][0]["field"] is None
    assert report["per_drug"][0]["issues"][0].get("message")
