# Lost in Simplification: Detecting Safety-Critical Errors in LLM-Generated Medication Instructions

This repository contains the code and frozen evaluation artifacts for our NeurIPS 2026–style paper on simplifying FDA drug labels with open-weight LLMs. The pipeline pulls structured labels from [openFDA](https://open.fda.gov/), runs simplification (Groq or OpenAI), extracts safety-critical fields, compares originals to simplifications with heuristics, scores preservation with an LLM judge, aggregates results, and computes readability metrics.

## Authors

- **Rohit Kulkarni** — co-author (pipeline, evaluation, writing).
- **Atharv Talnikar** — co-author (experiments, reproducibility scripts, paper tables).
- **Christopher Huitt** — co-author (analysis, writing).

All authors contributed to implementation and the manuscript; roles above are approximate.

## Repository layout

```text
.
├── requirements.txt
├── .gitignore                # Ignores secrets, venv, large/regenerable JSON
├── .env                      # Not in git: create locally (see Setup)
├── paper.pdf                 # Final paper (NeurIPS 2026 format)
├── scripts/
│   ├── ingest_data.py
│   ├── simplify_labels.py
│   ├── extract_labels.py
│   ├── compare_extractions.py
│   ├── evaluate_labels.py
│   ├── analyze_results.py
│   ├── validate_judge.py
│   ├── compute_readability.py
│   ├── run_pipeline.py
│   ├── run_full_evaluation.sh
│   ├── generate_paper_tables.py
│   └── test_*.py
├── data/
│   ├── drug_labels.json              # Frozen ground truth (force-tracked)
│   └── raw_labels/sample_labels.json   # Small fixture
└── outputs/
    ├── paper_tables.json               # Aggregated numbers for tables / TikZ
    ├── manual_validation.json          # Human labels for judge validation
    ├── validation_results.json         # Output of validate_judge.py
    ├── runs/<run_name>/                # One directory per simplifier run
    │   ├── evaluation_report.json      # Tracked (small)
    │   ├── analysis_results.json
    │   ├── comparison_report.json
    │   ├── readability.json
    │   ├── simplified_labels.json      # Gitignored (large)
    │   ├── extracted_original.json     # Gitignored
    │   └── extracted_simplified.json   # Gitignored
    └── evaluation_report.json          # Legacy single-run copies (older layout)
```

Canonical artifacts for the paper’s two models live under `outputs/runs/gpt-oss/` and `outputs/runs/llama-70b/`. The JSON files at `outputs/` root (without `runs/`) are from an earlier single-directory layout and are kept for reference.

## Setup

1. Clone the repository and enter the project root.

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the **repository root** (it is gitignored). Required for the full evaluation script and Groq-backed steps:

   - **`GROQ_API_KEY`** — simplification when using `--provider groq`, and the LLM judge (Groq) in `run_full_evaluation.sh`.

   Optional:

   - **`OPENFDA_API_KEY`** — higher rate limits for `scripts/ingest_data.py` (the script loads `.env` from the repo root automatically).
   - **`OPENAI_API_KEY`** — if you use `--provider openai` for simplification.
   - **`JUDGE_MODEL`** — override judge default (`llama-3.1-8b-instant`) used by `run_full_evaluation.sh`.

If this repo sits under a parent folder that already has a `.env` (e.g. `GENAI/.env`), `run_full_evaluation.sh` sources the repo `.env` first, then `../.env`.

## Full pipeline (one simplifier run)

From the repo root, after `data/drug_labels.json` exists (run ingest once if missing):

```bash
chmod +x scripts/run_full_evaluation.sh
./scripts/run_full_evaluation.sh <run_name> <provider> <model>
```

Examples matching the committed run directories:

```bash
./scripts/run_full_evaluation.sh gpt-oss groq openai/gpt-oss-120b
./scripts/run_full_evaluation.sh llama-70b groq llama-3.3-70b-versatile
```

Artifacts are written to `outputs/runs/<run_name>/`. Large intermediates (`simplified_labels.json`, `extracted_*.json`) are gitignored; commit the evaluation, comparison, analysis, and readability JSON if you need to snapshot a new run.

## Paper tables JSON

From the repo root (requires both runs to have `evaluation_report.json`, `comparison_report.json`, and `readability.json`):

```bash
python scripts/generate_paper_tables.py \
  --run1 outputs/runs/gpt-oss \
  --label1 "GPT-OSS-120B" \
  --run2 outputs/runs/llama-70b \
  --label2 "Llama-3.3-70B" \
  --output outputs/paper_tables.json
```

`--run1` is the primary run for judge/heuristic tables and the scatter coordinates.

## Tests

```bash
python -m pytest scripts/ -q
```

## Key results (paper-aligned)

- Across 25 drugs (three risk tiers), **38.7%** of safety-critical fields are judged **preserved** after simplification (primary run, GPT-OSS-120B).
- **Contraindications** and **drug interactions** are the weakest sections (judge-assigned drop rates **81.82%** and **90.48%** respectively on the primary run).
- **High-risk / controlled** medications show lower preservation (**28.9%** of fields preserved) than **simple OTC** drugs (**65.4%**).
- **Readability** improves by roughly **2.6–4.1** Flesch–Kincaid grade levels depending on model, but improvement does **not** track safety preservation; manual validation of the judge shows **~60%** agreement with Cohen’s **κ ≈ 0.27**.

## Manual validation of the LLM judge

`manual_validation.json` must be a **JSON array** of objects with `drug_name`, `field`, `human_judgment` (`PRESERVED`, `SOFTENED`, or `DROPPED`), and optional `reasoning`. The repository includes [`outputs/manual_validation.json`](outputs/manual_validation.json) as a reference.

Compare human labels to a run’s `evaluation_report.json` and write metrics:

```bash
python scripts/validate_judge.py \
  --human outputs/manual_validation.json \
  --judge outputs/runs/gpt-oss/evaluation_report.json \
  --output outputs/validation_results.json
```
