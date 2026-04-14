#!/usr/bin/env python3
"""
Paper-ready tables from two pipeline runs (primary = run1 for judge/heuristic/scatter).

Uncategorized drugs use TikZ tag `hr` (same as High-Risk/Controlled) to match a 3-series plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from analyze_results import drug_category, load_json, print_paper_summary, run_analysis
from judge_taxonomy import PRESERVED


def _mean_metric(rows: list[dict[str, Any]], side: str, key: str) -> tuple[float | None, int]:
    vals: list[float] = []
    for r in rows:
        block = r.get(side) or {}
        v = block.get(key)
        if v is not None:
            vals.append(float(v))
    if not vals:
        return None, 0
    return mean(vals), len(vals)


def readability_block(label: str, path: str) -> dict[str, Any]:
    data = load_json(path)
    rows = data.get("per_drug") or []
    fk_o, n1 = _mean_metric(rows, "original", "flesch_kincaid_grade")
    fk_s, n2 = _mean_metric(rows, "simplified", "flesch_kincaid_grade")
    fl_o, _ = _mean_metric(rows, "original", "flesch_reading_ease")
    fl_s, _ = _mean_metric(rows, "simplified", "flesch_reading_ease")

    worse_fk = 0
    worse_flesch = 0
    for r in rows:
        o = r["original"]["flesch_kincaid_grade"]
        s = r["simplified"]["flesch_kincaid_grade"]
        if o is not None and s is not None and float(s) > float(o):
            worse_fk += 1
        fo = r["original"]["flesch_reading_ease"]
        fs = r["simplified"]["flesch_reading_ease"]
        if fo is not None and fs is not None and float(fs) < float(fo):
            worse_flesch += 1

    delta_fk = (fk_s - fk_o) if fk_o is not None and fk_s is not None else None
    delta_flesch = (fl_s - fl_o) if fl_o is not None and fl_s is not None else None

    return {
        "label": label,
        "path": path,
        "n_drugs": len(rows),
        "flesch_reading_ease_original_mean": round(fl_o, 2) if fl_o is not None else None,
        "flesch_reading_ease_simplified_mean": round(fl_s, 2) if fl_s is not None else None,
        "flesch_reading_ease_delta_simplified_minus_original": round(delta_flesch, 2) if delta_flesch is not None else None,
        "flesch_kincaid_grade_original_mean": round(fk_o, 2) if fk_o is not None else None,
        "flesch_kincaid_grade_simplified_mean": round(fk_s, 2) if fk_s is not None else None,
        "flesch_kincaid_grade_delta_simplified_minus_original": round(delta_fk, 2) if delta_fk is not None else None,
        "n_drugs_worse_fk_grade_after_simplify": worse_fk,
        "n_drugs_worse_flesch_ease_after_simplify": worse_flesch,
    }


def preservation_rate_percent(evaluation: dict[str, Any], drug_name: str | None) -> float | None:
    if not drug_name:
        return None
    for row in evaluation.get("per_drug") or []:
        if str(row.get("drug_name", "")).strip() != str(drug_name).strip():
            continue
        js = row.get("judgments") or []
        if not js:
            return None
        p = sum(1 for j in js if str(j.get("judgment", "")).strip().upper() == PRESERVED)
        return 100.0 * p / len(js)
    return None


def tikz_tag(cat: str) -> str:
    if cat == "Simple OTC":
        return "otc"
    if cat == "Common Rx":
        return "rx"
    if cat in ("High-Risk/Controlled", "Uncategorized"):
        return "hr"
    return "hr"


def scatter_coordinates(
    evaluation: dict[str, Any], readability: dict[str, Any]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in readability.get("per_drug") or []:
        name = r.get("drug_name")
        fk_o = r["original"]["flesch_kincaid_grade"]
        fk_s = r["simplified"]["flesch_kincaid_grade"]
        if fk_o is None or fk_s is None:
            fk_improve = None
        else:
            fk_improve = float(fk_o) - float(fk_s)
        pr = preservation_rate_percent(evaluation, str(name) if name else None)
        cat = drug_category(str(name) if name else "")
        tag = tikz_tag(cat)
        out.append(
            {
                "drug_name": name,
                "category": cat,
                "tikz_tag": tag,
                "fk_grade_improvement": round(fk_improve, 2) if fk_improve is not None else None,
                "preservation_rate_pct": round(pr, 1) if pr is not None else None,
            }
        )
    return out


def format_tikz_coordinates(points: list[dict[str, Any]]) -> dict[str, str]:
    groups: dict[str, list[tuple[float, float]]] = {"otc": [], "rx": [], "hr": []}
    for p in points:
        tag = p["tikz_tag"]
        fi = p.get("fk_grade_improvement")
        pr = p.get("preservation_rate_pct")
        if fi is None or pr is None:
            continue
        if tag not in groups:
            tag = "hr"
        groups[tag].append((float(fi), float(pr)))

    lines: dict[str, str] = {}
    for tag, coords in groups.items():
        # One \\addplot per tag; coordinates need no per-point class (scatter/classes sets marker).
        inner = " ".join(f"({x:.2f},{y:.0f})" for x, y in sorted(coords))
        lines[tag] = inner
    return lines


def print_readability_section(b1: dict[str, Any], label1: str, b2: dict[str, Any], label2: str) -> None:
    print("")
    print("=" * 72)
    print("READABILITY (cross-model; means over drugs, same ingest)")
    print("=" * 72)
    for b, lab in ((b1, label1), (b2, label2)):
        print(f"\n--- {lab} ({b['path']}) ---")
        print(f"  n_drugs: {b['n_drugs']}")
        print(
            f"  Flesch Reading Ease: {b['flesch_reading_ease_original_mean']} -> "
            f"{b['flesch_reading_ease_simplified_mean']} "
            f"(delta {b['flesch_reading_ease_delta_simplified_minus_original']})"
        )
        print(
            f"  FK grade level: {b['flesch_kincaid_grade_original_mean']} -> "
            f"{b['flesch_kincaid_grade_simplified_mean']} "
            f"(delta {b['flesch_kincaid_grade_delta_simplified_minus_original']})"
        )
        print(
            f"  Drugs worse after simplify (higher FK): {b['n_drugs_worse_fk_grade_after_simplify']}; "
            f"(lower Flesch): {b['n_drugs_worse_flesch_ease_after_simplify']}"
        )


def print_scatter_and_tikz(points: list[dict[str, Any]], tikz_flat: dict[str, str]) -> None:
    print("")
    print("=" * 72)
    print("SCATTER (run1): FK grade improvement vs preservation %")
    print("=" * 72)
    for p in sorted(points, key=lambda x: (x.get("drug_name") or "")):
        print(
            f"  {str(p.get('drug_name')):<48}  cat={p['category']:<22}  "
            f"fk_imp={p.get('fk_grade_improvement')}  pres%={p.get('preservation_rate_pct')}"
        )
    print("")
    print("=" * 72)
    print("TIKZ \\addplot coordinates (paste under scatter/classes otc/rx/hr)")
    print("=" * 72)
    for tag in ("otc", "rx", "hr"):
        body = tikz_flat.get(tag, "").strip()
        print(f"\n% {tag}")
        print(f"coordinates {{\n    {body}\n}};")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate paper tables from two run directories (run1 = primary for tables 1–3 and scatter)."
    )
    parser.add_argument("--run1", required=True, help="Path to primary run dir (e.g. outputs/runs/gpt-oss)")
    parser.add_argument("--label1", required=True, help='Display label for run1 (e.g. "GPT-OSS-120B")')
    parser.add_argument("--run2", required=True, help="Path to second run dir")
    parser.add_argument("--label2", required=True, help='Display label for run2')
    parser.add_argument(
        "--output",
        default="outputs/paper_tables.json",
        help="Write combined JSON artifact (default: outputs/paper_tables.json)",
    )
    args = parser.parse_args()

    run1 = Path(args.run1)
    run2 = Path(args.run2)
    ev1_path = run1 / "evaluation_report.json"
    cmp1_path = run1 / "comparison_report.json"
    r1_path = run1 / "readability.json"
    r2_path = run2 / "readability.json"

    for p in (ev1_path, cmp1_path, r1_path, r2_path):
        if not p.is_file():
            print(f"Missing required file: {p}", file=sys.stderr)
            return 1

    report = run_analysis(str(ev1_path), str(cmp1_path))
    t1 = report["table1_by_field"]
    t2 = report["table2_by_category"]
    t3 = report["table3_heuristic_vs_judge"]

    comp1 = load_json(str(cmp1_path))
    comp_summary = comp1.get("summary") or {}

    evaluation1 = load_json(str(ev1_path))
    readability1 = load_json(str(r1_path))
    readability2 = load_json(str(r2_path))

    rb1 = readability_block(args.label1, str(r1_path))
    rb2 = readability_block(args.label2, str(r2_path))

    scatter = scatter_coordinates(evaluation1, readability1)
    tikz_parts = format_tikz_coordinates(scatter)

    out_doc: dict[str, Any] = {
        "schema_version": "1",
        "run1": str(run1),
        "run2": str(run2),
        "label1": args.label1,
        "label2": args.label2,
        "table1_by_field": t1,
        "table2_by_category": t2,
        "table3_heuristic_vs_judge": t3,
        "comparison_summary_run1": {
            "total_issue_flags": comp_summary.get("total_issue_flags"),
            "original_records": comp_summary.get("original_records"),
            "compared_pairs": comp_summary.get("compared_pairs"),
        },
        "readability_run1": rb1,
        "readability_run2": rb2,
        "scatter_run1": scatter,
        "tikz_coordinates_by_tag": tikz_parts,
    }

    print_paper_summary(t1, t2, t3)
    print("")
    print("=" * 72)
    print(f"Heuristic flags (run1 comparison_report): total_issue_flags = {comp_summary.get('total_issue_flags')}")
    print("=" * 72)
    print_readability_section(rb1, args.label1, rb2, args.label2)
    print_scatter_and_tikz(scatter, tikz_parts)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2, ensure_ascii=False)
    print("")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
