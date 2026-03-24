"""
Heuristic comparison: aligned extracted_original vs extracted_simplified.

Flags (heuristic, not clinical truth):
  - DROPPED_FIELD: original has text, simplified is null/empty
  - POSSIBLE_DOSE_MISMATCH: numeric+unit tokens in original dosage not found in simplified dosage
  - POSSIBLE_CONTENT_LOSS: a long prefix of original field not found in simplified (substring check)
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

SAFETY_FIELDS = ["boxed_warning", "dosage", "warnings", "contraindications", "interactions"]

# Dose-like tokens: numbers with common units or frequency words
DOSE_TOKEN = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|hours?|hrs?|daily|twice|once|tid|bid|qid|q\d+h|tablet|tablets|capsule|capsules)\b",
    re.I,
)


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())


def _dose_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {m.group(0).strip().lower() for m in DOSE_TOKEN.finditer(text)}


def _significant_snippets(text: str | None, min_len: int = 60, max_snippets: int = 3) -> list[str]:
    if not text or len(text) < min_len:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for p in parts:
        t = p.strip()
        if len(t) >= 40:
            out.append(_norm(t)[:220])
        if len(out) >= max_snippets:
            break
    if not out and text:
        out.append(_norm(text)[:220])
    return out


def compare_pair(
    orig: dict[str, Any],
    simp: dict[str, Any],
) -> dict[str, Any]:
    drug = orig.get("drug_name") or simp.get("drug_name") or "UNKNOWN"
    issues: list[dict[str, str]] = []

    for field in SAFETY_FIELDS:
        o_raw = orig.get(field)
        s_raw = simp.get(field)
        o = str(o_raw).strip() if o_raw else ""
        s = str(s_raw).strip() if s_raw else ""

        if o and not s:
            issues.append({"type": "DROPPED_FIELD", "field": field, "detail": "original present, simplified empty"})

        if field == "dosage" and o and s:
            ot = _dose_tokens(o)
            st = _dose_tokens(s)
            missing = ot - st
            if missing:
                issues.append(
                    {
                        "type": "POSSIBLE_DOSE_MISMATCH",
                        "field": field,
                        "detail": f"tokens in original not in simplified: {sorted(missing)[:8]}",
                    }
                )

        if o and s and field != "dosage":
            no = _norm(o)
            ns = _norm(s)
            if len(no) > 80 and no[:200] not in ns:
                # Soft check: first substantial snippet
                snippets = _significant_snippets(o)
                for snip in snippets:
                    if snip and snip not in ns and len(snip) > 50:
                        issues.append(
                            {
                                "type": "POSSIBLE_CONTENT_LOSS",
                                "field": field,
                                "detail": "substantial original snippet not found in simplified",
                            }
                        )
                        break

    return {
        "drug_name": drug,
        "issue_count": len(issues),
        "issues": issues,
    }


def compare_files(path_original: str, path_simplified: str) -> dict[str, Any]:
    with open(path_original, "r", encoding="utf-8") as f:
        original_list = json.load(f)
    with open(path_simplified, "r", encoding="utf-8") as f:
        simplified_list = json.load(f)

    by_drug: dict[str, dict[str, Any]] = {}
    for row in simplified_list:
        dn = row.get("drug_name")
        if dn:
            by_drug[str(dn).upper()] = row

    per_drug: list[dict[str, Any]] = []
    missing_match = 0
    for row in original_list:
        key = str(row.get("drug_name", "")).upper()
        if key not in by_drug:
            missing_match += 1
            per_drug.append(
                {
                    "drug_name": row.get("drug_name"),
                    "issue_count": 1,
                    "issues": [{"type": "NO_MATCH", "field": "—", "detail": "no simplified record with same drug_name"}],
                }
            )
            continue
        per_drug.append(compare_pair(row, by_drug[key]))

    total_issues = sum(r["issue_count"] for r in per_drug)
    return {
        "summary": {
            "original_records": len(original_list),
            "simplified_records": len(simplified_list),
            "compared_pairs": len(original_list) - missing_match,
            "unmatched_original": missing_match,
            "total_issue_flags": total_issues,
        },
        "per_drug": per_drug,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare extracted original vs extracted simplified JSON.")
    parser.add_argument("--original", default="extracted_original.json", help="Path to extracted_original.json")
    parser.add_argument("--simplified", default="extracted_simplified.json", help="Path to extracted_simplified.json")
    parser.add_argument("--output", default="comparison_report.json", help="Write full report JSON here")
    args = parser.parse_args()

    report = compare_files(args.original, args.simplified)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    s = report["summary"]
    print("--- Comparison summary ---")
    print(f"Original records:      {s['original_records']}")
    print(f"Simplified records:    {s['simplified_records']}")
    print(f"Matched pairs:         {s['compared_pairs']}")
    print(f"Unmatched original:    {s['unmatched_original']}")
    print(f"Total flagged issues:  {s['total_issue_flags']}")
    print(f"Report written to:     {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
