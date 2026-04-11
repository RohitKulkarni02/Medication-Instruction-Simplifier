#!/usr/bin/env python3
"""
Rohit batch: run simplification twice on Groq with two different models (same GROQ_API_KEY),
then optional readability reports per run. Outputs live under outputs/runs/ (gitignored).

Default pairing (both are Groq model IDs — no api.openai.com key):
  - OpenAI GPT-OSS (open-weight) on Groq: openai/gpt-oss-120b
  - Meta Llama on Groq: llama-3.3-70b-versatile

Example:
  python3 scripts/run_dual_model_simplification.py \\
    --ingest data/drug_labels.json \\
    --sleep-ms 2500

Requires: GROQ_API_KEY in .env only.

Skip either run: --skip-1 / --skip-2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Groq-hosted only. Run 1 = OpenAI family (GPT-OSS), run 2 = Llama (not paid OpenAI API).
DEFAULT_MODEL_1 = "openai/gpt-oss-120b"
DEFAULT_MODEL_2 = "llama-3.3-70b-versatile"


def _py() -> str:
    return sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch: GPT-OSS (on Groq) + Llama (on Groq), same GROQ_API_KEY — outputs/runs/* + readability."
    )
    parser.add_argument("--ingest", default="data/drug_labels.json", help="Ingest JSON array path.")
    parser.add_argument(
        "--out-root",
        default="outputs/runs",
        help="Directory for per-run subfolders (see --out-label-1 / --out-label-2).",
    )
    parser.add_argument(
        "--groq-model-1",
        default=os.environ.get("GROQ_SIMPLIFY_MODEL", DEFAULT_MODEL_1),
        help="Run 1: Groq chat model id (default: env GROQ_SIMPLIFY_MODEL or openai/gpt-oss-120b). "
        "Smaller/faster GPT-class on Groq: openai/gpt-oss-20b.",
    )
    parser.add_argument(
        "--groq-model-2",
        default=os.environ.get("GROQ_SIMPLIFY_MODEL_2", DEFAULT_MODEL_2),
        help="Run 2: Llama (or other) Groq model id (default: env GROQ_SIMPLIFY_MODEL_2 or llama-3.3-70b-versatile).",
    )
    parser.add_argument(
        "--out-label-1",
        default="groq_gpt_oss",
        help="Subfolder under --out-root for run 1 (default groq_gpt_oss).",
    )
    parser.add_argument(
        "--out-label-2",
        default="groq_llama",
        help="Subfolder under --out-root for run 2 (default groq_llama).",
    )
    parser.add_argument("--sleep-ms", type=int, default=2500)
    # Keeps prompt+max_tokens under Groq free-tier per-request limits when paired with completion cap.
    parser.add_argument("--max-llm-chars", type=int, default=4200)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=0,
        help="Forward to simplify_labels.py (0 = its default: env SIMPLIFY_MAX_OUTPUT_TOKENS or 8192).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--skip-1", action="store_true", help="Skip first Groq run.")
    parser.add_argument("--skip-2", action="store_true", help="Skip second Groq run.")
    parser.add_argument("--skip-readability", action="store_true")
    args = parser.parse_args()

    ingest = Path(args.ingest)
    if not ingest.is_absolute():
        ingest = ROOT / ingest
    if not ingest.is_file():
        print(f"Missing ingest file: {ingest}", file=sys.stderr)
        return 1

    def rel_to_root(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return str(p.resolve())

    out_root = ROOT / args.out_root
    dir1 = out_root / args.out_label_1
    dir2 = out_root / args.out_label_2
    json1 = dir1 / "simplified_labels.json"
    json2 = dir2 / "simplified_labels.json"

    simplify = [_py(), "scripts/simplify_labels.py", "--input", rel_to_root(ingest)]

    runs: list[tuple[str, Path, Path, str]] = []
    if not args.skip_1:
        runs.append(("1", dir1, json1, args.groq_model_1))
    if not args.skip_2:
        runs.append(("2", dir2, json2, args.groq_model_2))

    for _label, out_dir, out_json, model in runs:
        out_dir.mkdir(parents=True, exist_ok=True)
        run(
            simplify
            + [
                "--provider",
                "groq",
                "--model",
                model,
                "--output",
                rel_to_root(out_json),
                "--pretty",
                "--sleep-ms",
                str(args.sleep_ms),
                "--max-llm-chars",
                str(args.max_llm_chars),
                "--temperature",
                str(args.temperature),
            ]
            + (["--max-output-tokens", str(args.max_output_tokens)] if args.max_output_tokens > 0 else [])
        )
        meta = {"provider": "groq", "model": model}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if not args.skip_readability:
        read = [_py(), "scripts/compute_readability.py", "--ingest", rel_to_root(ingest)]
        if not args.skip_1 and json1.is_file():
            run(
                read
                + [
                    "--simplified",
                    rel_to_root(json1),
                    "--output",
                    rel_to_root(dir1 / "readability.json"),
                ]
            )
        if not args.skip_2 and json2.is_file():
            run(
                read
                + [
                    "--simplified",
                    rel_to_root(json2),
                    "--output",
                    rel_to_root(dir2 / "readability.json"),
                ]
            )

    print("\nDone (both runs use GROQ_API_KEY).")
    if not args.skip_1:
        print(f"  Run 1 ({args.groq_model_1}): {json1}")
    if not args.skip_2:
        print(f"  Run 2 ({args.groq_model_2}): {json2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
