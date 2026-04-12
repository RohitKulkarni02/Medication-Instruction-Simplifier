from __future__ import annotations

import json
from pathlib import Path

import pytest

from judge_taxonomy import DROPPED, PRESERVED, SOFTENED
from validate_judge import cohens_kappa_multiclass, confusion_3x3, run_validation


def test_cohens_kappa_perfect_agreement():
    pairs = [(PRESERVED, PRESERVED), (SOFTENED, SOFTENED), (DROPPED, DROPPED)]
    k, note = cohens_kappa_multiclass(pairs)
    assert note == "ok"
    assert k is not None and k > 0.99


def test_cohens_kappa_no_pair():
    k, note = cohens_kappa_multiclass([])
    assert k is None


def test_confusion_matrix():
    pairs = [
        (PRESERVED, SOFTENED),
        (PRESERVED, PRESERVED),
        (SOFTENED, SOFTENED),
    ]
    m = confusion_3x3(pairs)
    assert m[PRESERVED][PRESERVED] == 1
    assert m[PRESERVED][SOFTENED] == 1


def test_run_validation_agreement_and_disagreement(tmp_path: Path):
    human = [
        {"drug_name": "IBUPROFEN", "field": "dosage", "human_judgment": "PRESERVED"},
        {"drug_name": "IBUPROFEN", "field": "warnings", "human_judgment": "DROPPED"},
    ]
    judge = {
        "per_drug": [
            {
                "drug_name": "IBUPROFEN",
                "judgments": [
                    {"field": "dosage", "judgment": "PRESERVED", "explanation": "ok"},
                    {"field": "warnings", "judgment": "PRESERVED", "explanation": "judge says ok"},
                ],
            }
        ]
    }
    hp = tmp_path / "h.json"
    jp = tmp_path / "j.json"
    hp.write_text(json.dumps(human), encoding="utf-8")
    jp.write_text(json.dumps(judge), encoding="utf-8")
    rep = run_validation(str(hp), str(jp))
    assert rep["n_matched_pairs"] == 2
    assert rep["n_agreements"] == 1
    assert rep["agreement_rate_pct"] == 50.0
    assert len(rep["disagreements"]) == 1
    assert rep["disagreements"][0]["field"] == "warnings"


def test_run_validation_missing_judge_row(tmp_path: Path):
    human = [{"drug_name": "X", "field": "dosage", "human_judgment": "PRESERVED"}]
    judge = {"per_drug": []}
    hp = tmp_path / "h.json"
    jp = tmp_path / "j.json"
    hp.write_text(json.dumps(human), encoding="utf-8")
    jp.write_text(json.dumps(judge), encoding="utf-8")
    rep = run_validation(str(hp), str(jp))
    assert rep["n_matched_pairs"] == 0
    assert rep["n_missing_judge_or_invalid"] == 1
