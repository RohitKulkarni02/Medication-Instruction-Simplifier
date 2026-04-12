"""
Aggregate evaluation + comparison reports (default paths under `outputs/`) for paper tables.

No API calls — pure local JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

from judge_taxonomy import DROPPED, PRESERVED, SOFTENED

ERROR = "ERROR"

FIELDS = ["boxed_warning", "dosage", "warnings", "contraindications", "interactions"]

# Longer keys first so e.g. ACETAMINOPHEN matches before ASPIRIN in combo products.
SIMPLE_OTC_KEYS = sorted(
    [
        "IBUPROFEN",
        "ACETAMINOPHEN",
        "DIPHENHYDRAMINE",
        "LORATADINE",
        "OMEPRAZOLE",
        "CETIRIZINE",
        "ASPIRIN",
        "NAPROXEN",
    ],
    key=len,
    reverse=True,
)
COMMON_RX_KEYS = sorted(
    [
        "AMOXICILLIN",
        "METFORMIN",
        "LISINOPRIL",
        "ATORVASTATIN",
        "METOPROLOL",
        "SERTRALINE",
        "PREDNISONE",
        "AZITHROMYCIN",
    ],
    key=len,
    reverse=True,
)
HIGH_RISK_KEYS = sorted(
    [
        "METHOTREXATE",
        "TOFACITINIB",
        "WARFARIN",
        "ISOTRETINOIN",
        "CLOZAPINE",
        "OXYCODONE",
        "ALPRAZOLAM",
        "AMPHETAMINE",
        "DIAZEPAM",
    ],
    key=len,
    reverse=True,
)


def drug_category(drug_name: str) -> str:
    """Map openFDA-style drug_name to one of three paper categories or Uncategorized."""
    n = (drug_name or "").strip().upper()
    if not n:
        return "Uncategorized"

    def matches(keys: list[str]) -> bool:
        for k in keys:
            if n == k or n.startswith(k + " ") or n.startswith(k + ","):
                return True
        return False

    if matches(HIGH_RISK_KEYS):
        return "High-Risk/Controlled"
    if matches(COMMON_RX_KEYS):
        return "Common Rx"
    if matches(SIMPLE_OTC_KEYS):
        return "Simple OTC"
    return "Uncategorized"


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def table1_by_field(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """Counts and percentages of PRESERVED/SOFTENED/DROPPED/ERROR per field."""
    counts: dict[str, dict[str, int]] = {f: defaultdict(int) for f in FIELDS}
    for row in evaluation.get("per_drug") or []:
        for j in row.get("judgments") or []:
            field = j.get("field")
            if field not in counts:
                continue
            verdict = str(j.get("judgment", "")).strip().upper()
            if verdict in (PRESERVED, SOFTENED, DROPPED, ERROR):
                counts[field][verdict] += 1

    out: list[dict[str, Any]] = []
    for field in FIELDS:
        c = counts[field]
        total = sum(c.values())
        row: dict[str, Any] = {"field": field, "total_judgments": total}
        for v in (PRESERVED, SOFTENED, DROPPED, ERROR):
            n = int(c.get(v, 0))
            row[v.lower()] = n
            row[f"{v.lower()}_pct"] = round(100.0 * n / total, 2) if total else 0.0
        out.append(row)
    return out


def table2_by_category(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate judgments by drug category."""
    cat_counts: dict[str, dict[str, int]] = {
        "Simple OTC": defaultdict(int),
        "Common Rx": defaultdict(int),
        "High-Risk/Controlled": defaultdict(int),
        "Uncategorized": defaultdict(int),
    }
    cat_drugs: dict[str, int] = defaultdict(int)

    for row in evaluation.get("per_drug") or []:
        drug = row.get("drug_name") or "UNKNOWN"
        cat = drug_category(str(drug))
        cat_drugs[cat] += 1
        for j in row.get("judgments") or []:
            verdict = str(j.get("judgment", "")).strip().upper()
            if verdict in (PRESERVED, SOFTENED, DROPPED, ERROR):
                cat_counts[cat][verdict] += 1

    order = ["Simple OTC", "Common Rx", "High-Risk/Controlled", "Uncategorized"]
    out: list[dict[str, Any]] = []
    for cat in order:
        c = cat_counts[cat]
        total = sum(c.values())
        r: dict[str, Any] = {
            "category": cat,
            "drugs_in_category": cat_drugs[cat],
            "total_judgments": total,
        }
        for v in (PRESERVED, SOFTENED, DROPPED, ERROR):
            n = int(c.get(v, 0))
            r[v.lower()] = n
            r[f"{v.lower()}_pct"] = round(100.0 * n / total, 2) if total else 0.0
        out.append(r)
    return out


def heuristic_flagged_for_field(issues: list[dict[str, Any]], field: str) -> bool:
    for iss in issues or []:
        if iss.get("type") == "NO_MATCH":
            continue
        f = str(iss.get("field", "")).strip()
        if f == field:
            return True
    return False


def build_heuristic_index(comparison: dict[str, Any]) -> dict[tuple[str, str], bool]:
    """(drug_name_upper, field) -> flagged."""
    idx: dict[tuple[str, str], bool] = {}
    for row in comparison.get("per_drug") or []:
        drug = str(row.get("drug_name", "")).strip().upper()
        flagged: dict[str, bool] = {f: False for f in FIELDS}
        for iss in row.get("issues") or []:
            f = str(iss.get("field", "")).strip()
            if f in FIELDS and iss.get("type") != "NO_MATCH":
                flagged[f] = True
        for f in FIELDS:
            idx[(drug, f)] = flagged[f]
    return idx


def build_judge_index(evaluation: dict[str, Any]) -> dict[tuple[str, str], str]:
    """(drug_name_upper, field) -> judgment uppercase."""
    idx: dict[tuple[str, str], str] = {}
    for row in evaluation.get("per_drug") or []:
        drug = str(row.get("drug_name", "")).strip().upper()
        for j in row.get("judgments") or []:
            f = str(j.get("field", "")).strip()
            if f not in FIELDS:
                continue
            idx[(drug, f)] = str(j.get("judgment", "")).strip().upper()
    return idx


def table3_heuristic_vs_judge(
    evaluation: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    heur = build_heuristic_index(comparison)
    judge = build_judge_index(evaluation)

    agreement = false_alarm = heuristic_miss = both_ok = 0
    heur_flag_judge_error = heur_clear_judge_error = 0
    judge_problem = {SOFTENED, DROPPED}

    keys = sorted(set(heur.keys()) & set(judge.keys()))
    for key in keys:
        h_flag = heur[key]
        j = judge[key]

        if j == ERROR:
            if h_flag:
                heur_flag_judge_error += 1
            else:
                heur_clear_judge_error += 1
            continue

        if h_flag and j in judge_problem:
            agreement += 1
        elif h_flag and j == PRESERVED:
            false_alarm += 1
        elif (not h_flag) and j in judge_problem:
            heuristic_miss += 1
        elif (not h_flag) and j == PRESERVED:
            both_ok += 1

    total_classified = agreement + false_alarm + heuristic_miss + both_ok
    return {
        "pairs_with_both_heuristic_and_judge": len(keys),
        "agreement_heuristic_flagged_and_judge_softened_or_dropped": agreement,
        "false_alarm_heuristic_flagged_judge_preserved": false_alarm,
        "heuristic_miss_judge_softened_or_dropped": heuristic_miss,
        "both_clear_heuristic_ok_judge_preserved": both_ok,
        "heuristic_flagged_judge_error": heur_flag_judge_error,
        "heuristic_clear_judge_error": heur_clear_judge_error,
        "total_pairs_used_in_agreement_matrix": total_classified,
    }


def print_paper_summary(t1: list[dict], t2: list[dict], t3: dict[str, Any]) -> None:
    lines = [
        "",
        "=" * 72,
        "ANALYSIS SUMMARY (paste-friendly for paper)",
        "=" * 72,
        "",
        "Table 1 — Error rates by field (LLM judge counts and % of field total)",
        "-" * 72,
    ]
    for r in t1:
        lines.append(
            f"  {r['field']:<22} n={r['total_judgments']:<4}  "
            f"PRESERVED {r['preserved']} ({r['preserved_pct']}%)  "
            f"SOFTENED {r['softened']} ({r['softened_pct']}%)  "
            f"DROPPED {r['dropped']} ({r['dropped_pct']}%)  "
            f"ERROR {r.get('error', 0)} ({r.get('error_pct', 0.0)}%)"
        )
    lines.extend(
        [
            "",
            "Table 2 — Error rates by drug category (aggregated judgments)",
            "-" * 72,
        ]
    )
    for r in t2:
        if r["total_judgments"] == 0 and r["drugs_in_category"] == 0:
            continue
        lines.append(
            f"  {r['category']:<22} drugs={r['drugs_in_category']}  judgments={r['total_judgments']}  "
            f"P {r['preserved']} ({r['preserved_pct']}%)  "
            f"S {r['softened']} ({r['softened_pct']}%)  "
            f"D {r['dropped']} ({r['dropped_pct']}%)"
        )
    lines.extend(
        [
            "",
            "Table 3 — Heuristic compare vs LLM judge (same drug+field in both reports)",
            "-" * 72,
            f"  Pairs with both heuristic row and judge row: {t3['pairs_with_both_heuristic_and_judge']}",
            f"  Agreement (heuristic flagged, judge SOFTENED/DROPPED): {t3['agreement_heuristic_flagged_and_judge_softened_or_dropped']}",
            f"  False alarm (heuristic flagged, judge PRESERVED): {t3['false_alarm_heuristic_flagged_judge_preserved']}",
            f"  Heuristic miss (not flagged, judge SOFTENED/DROPPED): {t3['heuristic_miss_judge_softened_or_dropped']}",
            f"  Both clear (not flagged, judge PRESERVED): {t3['both_clear_heuristic_ok_judge_preserved']}",
            f"  Judge ERROR rows (excluded from 4-way matrix): "
            f"flagged+ERROR={t3['heuristic_flagged_judge_error']}, clear+ERROR={t3['heuristic_clear_judge_error']}",
            "",
            "=" * 72,
            "",
        ]
    )
    print("\n".join(lines), file=sys.stdout)


def run_analysis(evaluation_path: str, comparison_path: str) -> dict[str, Any]:
    evaluation = load_json(evaluation_path)
    comparison = load_json(comparison_path)

    t1 = table1_by_field(evaluation)
    t2 = table2_by_category(evaluation)
    t3 = table3_heuristic_vs_judge(evaluation, comparison)

    return {
        "schema_version": "1",
        "evaluation_path": evaluation_path,
        "comparison_path": comparison_path,
        "table1_by_field": t1,
        "table2_by_category": t2,
        "table3_heuristic_vs_judge": t3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate judge + heuristic reports for paper tables.")
    parser.add_argument("--evaluation", default="outputs/evaluation_report.json", help="Path to evaluation_report.json")
    parser.add_argument("--comparison", default="outputs/comparison_report.json", help="Path to comparison_report.json")
    parser.add_argument("--output", default="outputs/analysis_results.json", help="Write structured tables here")
    args = parser.parse_args()

    report = run_analysis(args.evaluation, args.comparison)
    print_paper_summary(report["table1_by_field"], report["table2_by_category"], report["table3_heuristic_vs_judge"])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
