"""
One-command orchestrator: ingest -> simplify -> extract (structured [+ text]) -> compare.

Run from repo root:
  python3 scripts/run_pipeline.py --drug ibuprofen
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _py() -> str:
    return sys.executable


def run_step(args_list: list[str]) -> None:
    print("+", " ".join(args_list))
    subprocess.run(args_list, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full medication label pipeline.")
    parser.add_argument("--drug", default="ibuprofen", help="Generic name for single-drug ingest.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingest; expect data/drug_labels.json.")
    parser.add_argument(
        "--simplify-provider",
        default="local",
        choices=["local", "openai", "groq"],
        help="Backend for simplify_labels.py (groq uses OpenAI SDK + GROQ_API_KEY).",
    )
    parser.add_argument(
        "--simplify-model",
        default=None,
        help="Model id (Groq or OpenAI; defaults from GROQ_SIMPLIFY_MODEL / OPENAI_MODEL if omitted).",
    )
    parser.add_argument(
        "--simplify-max-chars",
        type=int,
        default=0,
        help="Forward to simplify_labels.py --max-llm-chars (0 = script default / env).",
    )
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help="Also run simplified extraction from simplified_text only; write comparison_text report.",
    )
    args = parser.parse_args()

    os.chdir(ROOT)

    if not args.skip_ingest:
        run_step([_py(), "scripts/ingest_data.py", "--drug", args.drug, "--output", "drug_labels.json"])

    simp_cmd = [
        _py(),
        "scripts/simplify_labels.py",
        "--provider",
        args.simplify_provider,
        "--input",
        "data/drug_labels.json",
        "--output",
        "simplified_labels.json",
        "--pretty",
    ]
    if args.simplify_model:
        simp_cmd.extend(["--model", args.simplify_model])
    if args.simplify_max_chars > 0:
        simp_cmd.extend(["--max-llm-chars", str(args.simplify_max_chars)])
    run_step(simp_cmd)

    run_step(
        [
            _py(),
            "scripts/extract_labels.py",
            "--source",
            "original",
            "--input",
            "data/drug_labels.json",
            "--output",
            "extracted_original.json",
        ]
    )
    run_step(
        [
            _py(),
            "scripts/extract_labels.py",
            "--source",
            "simplified",
            "--input",
            "simplified_labels.json",
            "--output",
            "extracted_simplified.json",
            "--simplified-mode",
            "structured",
        ]
    )
    run_step(
        [
            _py(),
            "scripts/compare_extractions.py",
            "--original",
            "extracted_original.json",
            "--simplified",
            "extracted_simplified.json",
            "--output",
            "comparison_report.json",
        ]
    )

    if args.extract_text:
        run_step(
            [
                _py(),
                "scripts/extract_labels.py",
                "--source",
                "simplified",
                "--input",
                "simplified_labels.json",
                "--output",
                "extracted_simplified_from_text.json",
                "--simplified-mode",
                "from_text",
            ]
        )
        run_step(
            [
                _py(),
                "scripts/compare_extractions.py",
                "--original",
                "extracted_original.json",
                "--simplified",
                "extracted_simplified_from_text.json",
                "--output",
                "comparison_report_text_only.json",
            ]
        )

    print("\nDone. Outputs under:", ROOT)
    print("  data/drug_labels.json, simplified_labels.json")
    print("  extracted_original.json, extracted_simplified.json, comparison_report.json")
    if args.extract_text:
        print("  extracted_simplified_from_text.json, comparison_report_text_only.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
