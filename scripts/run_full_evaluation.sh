#!/usr/bin/env bash
# End-to-end pipeline for one simplifier model run. Artifacts under outputs/runs/<run_name>/
#
# Usage: ./scripts/run_full_evaluation.sh <run_name> <provider> <model>
# Example: ./scripts/run_full_evaluation.sh gpt-oss groq openai/gpt-oss-120b
#
# Requires: GROQ_API_KEY (judge uses Groq; use --provider groq for simplify to use the same key).
# Optional: JUDGE_MODEL (default: llama-3.1-8b-instant). For OpenAI simplify, set OPENAI_API_KEY.
# Simplification output budget: --max-output-tokens 0 defers to simplify_labels.py (default 8192, or
# SIMPLIFY_MAX_OUTPUT_TOKENS in .env). Do not use 2048 here — Groq JSON mode often needs more headroom.
# Paid Groq: raise GROQ_REQUEST_TOKEN_BUDGET in .env if you hit 413-style caps (see simplify_labels.py).
#
# Loads .env from the repo root, then ../.env (e.g. GENAI/.env when this repo lives under GENAI).
# Python scripts use python-dotenv; bash does not unless we source here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${#}" -ne 3 ]]; then
  echo "Usage: $0 <run_name> <provider> <model>" >&2
  echo "Example: $0 gpt-oss groq openai/gpt-oss-120b" >&2
  exit 1
fi

RUN_NAME="$1"
PROVIDER="$2"
MODEL="$3"
RUN_DIR="outputs/runs/${RUN_NAME}"
JUDGE_MODEL="${JUDGE_MODEL:-llama-3.1-8b-instant}"

PYTHON="${PYTHON:-python3}"

log() {
  echo "[$(date -Iseconds)] $*"
}

load_env_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$f"
    set +a
    log "Loaded environment from ${f}"
  fi
}

load_env_file "${REPO_ROOT}/.env"
load_env_file "${REPO_ROOT}/../.env"

fail() {
  echo "[$(date -Iseconds)] ERROR: $*" >&2
  exit 1
}

trap 'fail "Pipeline failed (bash line ${LINENO})"' ERR

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  fail "GROQ_API_KEY is not set (required for LLM judge on Groq)."
fi

if [[ "${PROVIDER}" == "openai" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  fail "OPENAI_API_KEY is not set but simplify --provider is openai."
fi

mkdir -p "${RUN_DIR}"

# --- 1. Ingest ---
if [[ -f data/drug_labels.json ]]; then
  log "Skipping ingest: data/drug_labels.json already exists."
else
  log "Running ingest -> data/drug_labels.json"
  "${PYTHON}" scripts/ingest_data.py --output data/drug_labels.json
fi

# --- 2. Simplify ---
log "Simplifying with provider=${PROVIDER} model=${MODEL}"
"${PYTHON}" scripts/simplify_labels.py \
  --provider "${PROVIDER}" \
  --model "${MODEL}" \
  --input data/drug_labels.json \
  --output "${RUN_DIR}/simplified_labels.json" \
  --pretty \
  --sleep-ms 3000 \
  --max-llm-chars 2000 \
  --max-output-tokens 0

# --- 3. Extract original ---
if [[ -f "${RUN_DIR}/extracted_original.json" ]]; then
  log "Skipping extract original: ${RUN_DIR}/extracted_original.json exists."
else
  log "Extracting original structured fields"
  "${PYTHON}" scripts/extract_labels.py --source original \
    --input data/drug_labels.json \
    --output "${RUN_DIR}/extracted_original.json"
fi

# --- 4. Extract simplified ---
log "Extracting simplified (structured)"
"${PYTHON}" scripts/extract_labels.py --source simplified \
  --input "${RUN_DIR}/simplified_labels.json" \
  --output "${RUN_DIR}/extracted_simplified.json" \
  --simplified-mode structured

# --- 5. Heuristic comparison ---
log "Heuristic comparison"
"${PYTHON}" scripts/compare_extractions.py \
  --original "${RUN_DIR}/extracted_original.json" \
  --simplified "${RUN_DIR}/extracted_simplified.json" \
  --output "${RUN_DIR}/comparison_report.json"

# --- 6. LLM judge ---
log "LLM judge (Groq, model=${JUDGE_MODEL})"
"${PYTHON}" scripts/evaluate_labels.py \
  --provider groq \
  --model "${JUDGE_MODEL}" \
  --original "${RUN_DIR}/extracted_original.json" \
  --simplified "${RUN_DIR}/extracted_simplified.json" \
  --output "${RUN_DIR}/evaluation_report.json" \
  --delay 1.0

# --- 7. Analysis ---
log "Aggregating analysis_results.json"
"${PYTHON}" scripts/analyze_results.py \
  --evaluation "${RUN_DIR}/evaluation_report.json" \
  --comparison "${RUN_DIR}/comparison_report.json" \
  --output "${RUN_DIR}/analysis_results.json"

# --- 8. Readability ---
log "Computing readability metrics"
"${PYTHON}" scripts/compute_readability.py \
  --ingest data/drug_labels.json \
  --simplified "${RUN_DIR}/simplified_labels.json" \
  --output "${RUN_DIR}/readability.json"

# --- 9. Summary ---
echo ""
echo "=== Run complete: ${RUN_NAME} ==="
echo "Artifacts in ${RUN_DIR}/"
"${PYTHON}" -c "
import json
from pathlib import Path
p = Path('${RUN_DIR}/evaluation_report.json')
with open(p) as f:
    r = json.load(f)
s = r['summary']
print(f\"Judgments: {s['total_judgments']}\")
print(f\"P:{s['preserved']} S:{s['softened']} D:{s['dropped']}\")
"
