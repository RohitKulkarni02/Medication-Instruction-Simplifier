"""
One-command orchestrator: ingest -> simplify -> extract (structured [+ text]) -> compare [-> evaluate].

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
        "--simplify-max-output-tokens",
        type=int,
        default=0,
        help="Forward to simplify_labels.py --max-output-tokens (0 = omit; script uses env/default).",
    )
    parser.add_argument(
        "--extract-text",
        action="store_true",
        help="Also run simplified extraction from simplified_text only; write comparison_text report.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="After compare: run evaluate_labels.py (LLM judge; requires API keys for --evaluate-provider).",
    )
    parser.add_argument(
        "--evaluate-provider",
        choices=["openai", "groq"],
        default="groq",
        help="Provider for evaluate_labels.py (default groq).",
    )
    parser.add_argument("--evaluate-model", default=None, help="Model id for judge (optional; uses script defaults).")
    parser.add_argument(
        "--evaluate-output",
        default="outputs/evaluation_report.json",
        help="Output path for evaluation_report.json (under outputs/ unless absolute).",
    )
    parser.add_argument(
        "--evaluate-delay",
        type=float,
        default=1.0,
        help="Seconds between judge API calls (forwarded to evaluate_labels.py --delay).",
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
        "outputs/simplified_labels.json",
        "--pretty",
    ]
    if args.simplify_model:
        simp_cmd.extend(["--model", args.simplify_model])
    if args.simplify_max_chars > 0:
        simp_cmd.extend(["--max-llm-chars", str(args.simplify_max_chars)])
    if args.simplify_max_output_tokens > 0:
        simp_cmd.extend(["--max-output-tokens", str(args.simplify_max_output_tokens)])
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
            "outputs/extracted_original.json",
        ]
    )
    run_step(
        [
            _py(),
            "scripts/extract_labels.py",
            "--source",
            "simplified",
            "--input",
            "outputs/simplified_labels.json",
            "--output",
            "outputs/extracted_simplified.json",
            "--simplified-mode",
            "structured",
        ]
    )
    run_step(
        [
            _py(),
            "scripts/compare_extractions.py",
            "--original",
            "outputs/extracted_original.json",
            "--simplified",
            "outputs/extracted_simplified.json",
            "--output",
            "outputs/comparison_report.json",
        ]
    )

    if args.evaluate:
        ev_cmd = [
            _py(),
            "scripts/evaluate_labels.py",
            "--original",
            "outputs/extracted_original.json",
            "--simplified",
            "outputs/extracted_simplified.json",
            "--output",
            args.evaluate_output,
            "--provider",
            args.evaluate_provider,
            "--delay",
            str(args.evaluate_delay),
        ]
        if args.evaluate_model:
            ev_cmd.extend(["--model", args.evaluate_model])
        run_step(ev_cmd)

    if args.extract_text:
        run_step(
            [
                _py(),
                "scripts/extract_labels.py",
                "--source",
                "simplified",
                "--input",
                "outputs/simplified_labels.json",
                "--output",
                "outputs/extracted_simplified_from_text.json",
                "--simplified-mode",
                "from_text",
            ]
        )
        run_step(
            [
                _py(),
                "scripts/compare_extractions.py",
                "--original",
                "outputs/extracted_original.json",
                "--simplified",
                "outputs/extracted_simplified_from_text.json",
                "--output",
                "outputs/comparison_report_text_only.json",
            ]
        )

    print("\nDone. Outputs under:", ROOT)
    print("  data/drug_labels.json, outputs/simplified_labels.json")
    print("  outputs/extracted_original.json, outputs/extracted_simplified.json, outputs/comparison_report.json")
    if args.evaluate:
        print(f"  {args.evaluate_output} (LLM judge)")
    if args.extract_text:
        print("  outputs/extracted_simplified_from_text.json, outputs/comparison_report_text_only.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
