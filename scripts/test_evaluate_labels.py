from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluate_labels import (
    ERROR,
    build_judge_user_prompt,
    call_judge_llm,
    evaluate_drug_pair,
    field_to_text,
    normalize_judgment_dict,
    parse_judge_json,
    recompute_summary,
)
from judge_taxonomy import DROPPED, PRESERVED, SOFTENED


def test_field_to_text_none_and_list():
    assert field_to_text(None) == ""
    assert field_to_text("") == ""
    assert field_to_text(["a", "b"]) == "a\nb"
    assert field_to_text("  x  ") == "x"


def test_skip_both_empty_no_judgment():
    orig = {"drug_name": "X", "boxed_warning": None, "dosage": "", "warnings": [], "contraindications": None, "interactions": None}
    simp = {
        "drug_name": "X",
        "boxed_warning": None,
        "dosage": "",
        "warnings": [],
        "contraindications": None,
        "interactions": None,
    }
    mock_llm = MagicMock()
    j, sb, sn = evaluate_drug_pair(
        orig,
        simp,
        MagicMock(),
        model_used="m",
        system="sys",
        temperature=0.0,
        max_output_tokens=100,
        delay=0.0,
        call_llm=mock_llm,
    )
    assert j == []
    assert sb == 5
    assert sn == 0
    mock_llm.assert_not_called()


def test_skip_no_original_increment():
    """Original empty but simplified present → skipped_no_original; never call LLM."""
    orig = {
        "drug_name": "X",
        "boxed_warning": None,
        "dosage": None,
        "warnings": [],
        "contraindications": None,
        "interactions": None,
    }
    simp = {
        "drug_name": "X",
        "boxed_warning": "has text",
        "dosage": None,
        "warnings": ["simp-only"],
        "contraindications": None,
        "interactions": None,
    }
    mock_llm = MagicMock()
    j, sb, sn = evaluate_drug_pair(
        orig,
        simp,
        MagicMock(),
        model_used="m",
        system="sys",
        temperature=0.0,
        max_output_tokens=100,
        delay=0.0,
        call_llm=mock_llm,
    )
    assert sn == 2
    assert sb == 3
    assert j == []
    mock_llm.assert_not_called()


def test_auto_dropped_when_simplified_empty():
    orig = {
        "drug_name": "IBU",
        "boxed_warning": "risk",
        "dosage": "10 mg",
        "warnings": "w",
        "contraindications": "c",
        "interactions": "i",
    }
    simp = {
        "drug_name": "IBU",
        "boxed_warning": "",
        "dosage": None,
        "warnings": [],
        "contraindications": " ",
        "interactions": None,
    }
    mock_llm = MagicMock()
    j, _, _ = evaluate_drug_pair(
        orig,
        simp,
        MagicMock(),
        model_used="m",
        system="sys",
        temperature=0.0,
        max_output_tokens=100,
        delay=0.0,
        call_llm=mock_llm,
    )
    fields = {x["field"]: x["judgment"] for x in j}
    assert fields["boxed_warning"] == DROPPED
    assert fields["dosage"] == DROPPED
    assert fields["warnings"] == DROPPED
    mock_llm.assert_not_called()


def test_build_judge_user_prompt_contains_parts():
    field = "warnings"
    o = "Original warning text about bleeding."
    s = "Simpler warning about bleeding risk."
    p = build_judge_user_prompt(field, o, s)
    assert field in p
    assert o in p
    assert s in p
    assert PRESERVED in p and SOFTENED in p and DROPPED in p


def test_parse_and_normalize_valid_json():
    raw = '{"judgment": "softened", "explanation": "Weaker wording."}'
    d = parse_judge_json(raw)
    n = normalize_judgment_dict(d)
    assert n["judgment"] == SOFTENED
    assert "Weaker" in n["explanation"]


def test_malformed_then_valid_call_judge_llm():
    """invoke_judge_completion_json_then_plain tries json_object then plain inside one call."""
    client = MagicMock()
    bad = MagicMock()
    bad.choices = [MagicMock(message=MagicMock(content="not json"))]
    good = MagicMock()
    good.choices = [MagicMock(message=MagicMock(content='{"judgment": "PRESERVED", "explanation": "ok"}'))]

    def create_side_effect(**kwargs):
        if kwargs.get("response_format"):
            return bad
        return good

    client.chat.completions.create.side_effect = create_side_effect
    norm, raw = call_judge_llm(
        client,
        model="m",
        temperature=0.0,
        max_output_tokens=64,
        system="sys",
        user="user",
    )
    assert norm["judgment"] == PRESERVED
    assert "ok" in norm["explanation"].lower() or norm["explanation"] == "ok"
    assert "PRESERVED" in raw or "preserved" in raw.lower()


def test_call_judge_llm_error_after_two_failures():
    client = MagicMock()
    bad = MagicMock()
    bad.choices = [MagicMock(message=MagicMock(content="not json"))]
    client.chat.completions.create.return_value = bad

    with pytest.raises(Exception):
        call_judge_llm(
            client,
            model="m",
            temperature=0.0,
            max_output_tokens=64,
            system="sys",
            user="user",
        )


def test_evaluate_drug_pair_error_judgment_on_llm_failure():
    orig = {
        "drug_name": "Z",
        "boxed_warning": "bw text here",
        "dosage": None,
        "contraindications": None,
        "warnings": None,
        "interactions": None,
    }
    simp = {
        "drug_name": "Z",
        "boxed_warning": "bw simplified ok",
        "dosage": None,
        "contraindications": None,
        "warnings": None,
        "interactions": None,
    }

    def boom(*_a, **_k):
        raise ValueError("parse failed")

    j, _, _ = evaluate_drug_pair(
        orig,
        simp,
        MagicMock(),
        model_used="m",
        system="sys",
        temperature=0.0,
        max_output_tokens=100,
        delay=0.0,
        call_llm=boom,
    )
    assert len(j) == 1
    assert j[0]["judgment"] == ERROR
    assert "parse failed" in j[0]["explanation"] or "malformed" in j[0]["explanation"].lower()


def test_recompute_summary_counts():
    per_drug = [
        {"drug_name": "A", "judgments": [{"judgment": PRESERVED}, {"judgment": SOFTENED}, {"judgment": ERROR}]},
        {"drug_name": "B", "judgments": [{"judgment": DROPPED}]},
    ]
    s = recompute_summary(per_drug, skipped_both_empty=2, skipped_no_original=1)
    assert s["total_judgments"] == 4
    assert s["preserved"] == 1
    assert s["softened"] == 1
    assert s["dropped"] == 1
    assert s["errors"] == 1
    assert s["skipped_both_empty"] == 2
    assert s["skipped_no_original"] == 1


def test_incremental_report_written(tmp_path):
    from evaluate_labels import write_report_atomic

    p = tmp_path / "out.json"
    write_report_atomic(str(p), {"schema_version": "1", "per_drug": []})
    assert p.read_text(encoding="utf-8").startswith("{")


def test_parse_judge_json_with_surrounding_text():
    raw = 'Here you go: {"judgment": "DROPPED", "explanation": "gone."} thanks'
    d = parse_judge_json(raw)
    assert normalize_judgment_dict(d)["judgment"] == DROPPED
