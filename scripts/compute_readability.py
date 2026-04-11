#!/usr/bin/env python3
"""
Readability scores: original (ingest) vs simplified text per drug.

Uses textstat: Flesch Reading Ease, Flesch-Kincaid grade, SMOG, Gunning Fog.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return data


def _original_blob(rec: dict[str, Any]) -> str:
    for key in ("full_label", "original_text", "label_text"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    parts: list[str] = []
    for key in (
        "boxed_warning",
        "dosage_and_administration",
        "warnings",
        "contraindications",
        "drug_interactions",
    ):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, list):
            parts.extend(str(x).strip() for x in v if str(x).strip())
    return "\n\n".join(parts)


def _simplified_blob(rec: dict[str, Any]) -> str:
    st = rec.get("simplified_text")
    if isinstance(st, str) and st.strip():
        return st.strip()
    parts: list[str] = []
    for key in ("boxed_warning", "dosage"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    for key in ("warnings", "contraindications", "interactions"):
        v = rec.get(key)
        if isinstance(v, list):
            parts.extend(str(x).strip() for x in v if str(x).strip())
        elif isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def _scores_for_text(text: str) -> dict[str, float | None]:
    if not text or len(text.strip()) < 20:
        return {
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "smog_index": None,
            "gunning_fog": None,
        }
    try:
        import textstat  # type: ignore
    except ImportError as e:
        raise SystemExit("Install textstat: pip install textstat") from e

    def safe(fn: Any) -> float | None:
        try:
            return float(fn(text))
        except Exception:
            return None

    return {
        "flesch_reading_ease": safe(textstat.flesch_reading_ease),
        "flesch_kincaid_grade": safe(textstat.flesch_kincaid_grade),
        "smog_index": safe(textstat.smog_index),
        "gunning_fog": safe(textstat.gunning_fog),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Readability metrics: ingest vs simplified JSON.")
    parser.add_argument("--ingest", required=True, help="Path to drug_labels.json (openFDA ingest output).")
    parser.add_argument("--simplified", required=True, help="Path to simplified_labels.json.")
    parser.add_argument("--output", required=True, help="Write JSON report here.")
    args = parser.parse_args()

    ingest = _load_json(args.ingest)
    simplified = _load_json(args.simplified)

    by_name: dict[str, dict[str, Any]] = {}
    for rec in ingest:
        name = (rec.get("drug_name") or rec.get("brand_name") or "").strip().upper()
        if name:
            by_name[name] = rec

    rows: list[dict[str, Any]] = []
    for srec in simplified:
        name = (srec.get("drug_name") or "").strip().upper()
        orec = by_name.get(name, {})
        orig_text = _original_blob(orec) if orec else ""
        sim_text = _simplified_blob(srec)

        orig_scores = _scores_for_text(orig_text)
        sim_scores = _scores_for_text(sim_text)

        delta: dict[str, float | None] = {}
        for k in orig_scores:
            a, b = orig_scores[k], sim_scores[k]
            delta[k] = (b - a) if a is not None and b is not None else None

        rows.append(
            {
                "drug_name": srec.get("drug_name"),
                "original_char_count": len(orig_text),
                "simplified_char_count": len(sim_text),
                "original": orig_scores,
                "simplified": sim_scores,
                "delta_simplified_minus_original": delta,
            }
        )

    out = {
        "ingest_path": args.ingest,
        "simplified_path": args.simplified,
        "n_drugs": len(rows),
        "per_drug": rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Wrote readability report for {len(rows)} drugs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
