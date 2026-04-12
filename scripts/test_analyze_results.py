from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_results import (
    drug_category,
    run_analysis,
    table1_by_field,
    table3_heuristic_vs_judge,
)


def test_drug_category_otc_rx_high():
    assert drug_category("IBUPROFEN") == "Simple OTC"
    assert drug_category("ACETAMINOPHEN, DEXTROMETHORPHAN HYDROBROMIDE, X") == "Simple OTC"
    assert drug_category("AMOXICILLIN") == "Common Rx"
    assert drug_category("ATORVASTATIN CALCIUM") == "Common Rx"
    assert drug_category("OXYCODONE") == "High-Risk/Controlled"
    assert drug_category("UNKNOWN DRUG") == "Uncategorized"


def test_table1_by_field_counts():
    evaluation = {
        "per_drug": [
            {
                "drug_name": "A",
                "judgments": [
                    {"field": "dosage", "judgment": "PRESERVED"},
                    {"field": "dosage", "judgment": "SOFTENED"},
                    {"field": "warnings", "judgment": "DROPPED"},
                ],
            }
        ]
    }
    t1 = table1_by_field(evaluation)
    by_field = {r["field"]: r for r in t1}
    assert by_field["dosage"]["total_judgments"] == 2
    assert by_field["dosage"]["preserved"] == 1
    assert by_field["dosage"]["softened"] == 1
    assert by_field["warnings"]["dropped"] == 1
    assert by_field["boxed_warning"]["total_judgments"] == 0


def test_table3_agreement_false_alarm_miss(tmp_path: Path):
    evaluation = {
        "per_drug": [
            {
                "drug_name": "X",
                "judgments": [
                    {"field": "dosage", "judgment": "DROPPED"},
                    {"field": "warnings", "judgment": "PRESERVED"},
                    {"field": "boxed_warning", "judgment": "SOFTENED"},
                ],
            }
        ]
    }
    comparison = {
        "per_drug": [
            {
                "drug_name": "X",
                "issues": [
                    {"type": "DROPPED_FIELD", "field": "dosage", "detail": "x"},
                    {"type": "POSSIBLE_CONTENT_LOSS", "field": "warnings", "detail": "x"},
                    {"type": "POSSIBLE_DOSE_MISMATCH", "field": "boxed_warning", "detail": "x"},
                ],
            }
        ]
    }
    t3 = table3_heuristic_vs_judge(evaluation, comparison)
    assert t3["agreement_heuristic_flagged_and_judge_softened_or_dropped"] == 2  # dosage+boxed
    assert t3["false_alarm_heuristic_flagged_judge_preserved"] == 1  # warnings


def test_run_analysis_writes_roundtrip(tmp_path: Path):
    ev = {
        "per_drug": [
            {
                "drug_name": "IBUPROFEN",
                "judgments": [{"field": "dosage", "judgment": "PRESERVED"}],
            }
        ]
    }
    comp = {"per_drug": [{"drug_name": "IBUPROFEN", "issues": []}]}
    ev_path = tmp_path / "ev.json"
    co_path = tmp_path / "co.json"
    ev_path.write_text(json.dumps(ev), encoding="utf-8")
    co_path.write_text(json.dumps(comp), encoding="utf-8")
    rep = run_analysis(str(ev_path), str(co_path))
    assert "table1_by_field" in rep
    assert rep["table3_heuristic_vs_judge"]["both_clear_heuristic_ok_judge_preserved"] == 1
