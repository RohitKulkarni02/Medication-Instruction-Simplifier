"""
Compare hand labels (manual_validation.json) to LLM judge output.

No API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from judge_taxonomy import DROPPED, PRESERVED, SOFTENED

ERROR = "ERROR"
VALID = frozenset({PRESERVED, SOFTENED, DROPPED, ERROR})


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def judge_lookup(evaluation: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """(drug_upper, field_upper) -> judgment row."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in evaluation.get("per_drug") or []:
        drug = str(row.get("drug_name", "")).strip().upper()
        for j in row.get("judgments") or []:
            field = str(j.get("field", "")).strip().upper()
            out[(drug, field)] = j
    return out


def cohens_kappa_multiclass(
    pairs: list[tuple[str, str]],
) -> tuple[float | None, str]:
    """
    Cohen's kappa for two raters on the same N items, 3+ categories.
    pairs: list of (human_label, judge_label) both uppercase.
    """
    if not pairs:
        return None, "no_pairs"

    cats = sorted(set(l for p in pairs for l in p))
    n = len(pairs)
    p_o = sum(1 for h, j in pairs if h == j) / n

    # Marginal counts
    nh: dict[str, int] = {c: 0 for c in cats}
    nj: dict[str, int] = {c: 0 for c in cats}
    for h, j in pairs:
        nh[h] = nh.get(h, 0) + 1
        nj[j] = nj.get(j, 0) + 1

    p_e = sum((nh[c] / n) * (nj[c] / n) for c in cats)
    denom = 1.0 - p_e
    if denom <= 1e-12:
        return None, "chance_agreement_1"
    return (p_o - p_e) / denom, "ok"


def confusion_3x3(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """human -> judge counts for PRESERVED/SOFTENED/DROPPED only."""
    mat: dict[str, dict[str, int]] = {
        PRESERVED: {PRESERVED: 0, SOFTENED: 0, DROPPED: 0},
        SOFTENED: {PRESERVED: 0, SOFTENED: 0, DROPPED: 0},
        DROPPED: {PRESERVED: 0, SOFTENED: 0, DROPPED: 0},
    }
    for h, j in pairs:
        if h in mat and j in mat[h]:
            mat[h][j] += 1
    return mat


def run_validation(human_path: str, judge_path: str) -> dict[str, Any]:
    human_rows = load_json(human_path)
    if not isinstance(human_rows, list):
        raise ValueError("manual_validation.json must be a JSON array")

    evaluation = load_json(judge_path)
    jmap = judge_lookup(evaluation)

    matched: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []

    for item in human_rows:
        drug = str(item.get("drug_name", "")).strip().upper()
        field = str(item.get("field", "")).strip().upper()
        hum = str(item.get("human_judgment", "")).strip().upper()
        if hum not in (PRESERVED, SOFTENED, DROPPED):
            missing.append({"reason": "invalid_human_judgment", "item": item})
            continue

        jrow = jmap.get((drug, field))
        if not jrow:
            missing.append({"reason": "no_judge_row", "drug_name": item.get("drug_name"), "field": item.get("field")})
            continue

        jver = str(jrow.get("judgment", "")).strip().upper()
        rec = {
            "drug_name": item.get("drug_name"),
            "field": item.get("field"),
            "human_judgment": hum,
            "judge_judgment": jver,
            "judge_explanation": jrow.get("explanation", ""),
        }
        matched.append(rec)
        if hum != jver:
            disagreements.append(rec)

    n = len(matched)
    agreements = sum(1 for r in matched if r["human_judgment"] == r["judge_judgment"])
    agreement_pct = round(100.0 * agreements / n, 2) if n else 0.0

    # Kappa: only 3-way P/S/D; exclude judge ERROR or unknown
    kappa_pairs: list[tuple[str, str]] = []
    excluded_judge_error = 0
    for r in matched:
        h = r["human_judgment"]
        j = r["judge_judgment"]
        if j == ERROR:
            excluded_judge_error += 1
            continue
        if j not in (PRESERVED, SOFTENED, DROPPED):
            excluded_judge_error += 1
            continue
        kappa_pairs.append((h, j))

    kappa, kappa_note = cohens_kappa_multiclass(kappa_pairs)
    conf = confusion_3x3(kappa_pairs)

    return {
        "schema_version": "1",
        "human_path": human_path,
        "judge_path": judge_path,
        "n_human_rows": len(human_rows),
        "n_matched_pairs": n,
        "n_missing_judge_or_invalid": len(missing),
        "n_agreements": agreements,
        "agreement_rate_pct": agreement_pct,
        "cohens_kappa": kappa,
        "cohens_kappa_note": kappa_note,
        "n_excluded_from_kappa_judge_not_three_way": excluded_judge_error,
        "confusion_matrix_human_rows_judge_cols": conf,
        "disagreements": disagreements,
        "missing_or_invalid": missing,
    }


def print_summary(rep: dict[str, Any]) -> None:
    print("\n" + "=" * 60, file=sys.stdout)
    print("VALIDATION: human vs LLM judge", file=sys.stdout)
    print("=" * 60, file=sys.stdout)
    print(f"  Matched pairs:        {rep['n_matched_pairs']}", file=sys.stdout)
    print(f"  Agreement rate:       {rep['agreement_rate_pct']}%", file=sys.stdout)
    print(f"  Cohen's kappa:        {rep['cohens_kappa']!s} ({rep['cohens_kappa_note']})", file=sys.stdout)
    print(f"  Excluded from kappa:  {rep['n_excluded_from_kappa_judge_not_three_way']} (judge not P/S/D)", file=sys.stdout)
    print(f"  Disagreements:        {len(rep['disagreements'])}", file=sys.stdout)
    print("=" * 60 + "\n", file=sys.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate human labels against evaluation_report.json.")
    parser.add_argument("--human", required=True, help="Path to manual_validation.json")
    parser.add_argument("--judge", default="outputs/evaluation_report.json", help="Path to evaluation_report.json")
    parser.add_argument("--output", default="outputs/validation_results.json", help="Write results JSON here")
    args = parser.parse_args()

    try:
        rep = run_validation(args.human, args.judge)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print_summary(rep)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.output}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
