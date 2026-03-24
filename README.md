# Medication-Instruction-Simplifier

**CS6180 · Group 14**

End-to-end tooling to fetch FDA drug labels from [openFDA](https://open.fda.gov/), optionally simplify them with an LLM (or a local reformatting path), extract structured safety sections, and compare original vs simplified extractions with lightweight heuristics.

---

## Overview

| Stage | Script | Role |
|--------|--------|------|
| Ingest | `scripts/ingest_data.py` | Query openFDA, normalize label fields, export JSON or CSV |
| Simplify | `scripts/simplify_labels.py` | Produce patient-friendly text and structured fields (local, OpenAI, or Groq) |
| Extract | `scripts/extract_labels.py` | Map raw or simplified records into a unified safety-field schema |
| Compare | `scripts/compare_extractions.py` | Flag possible drops / dose mismatches / content loss between extractions |
| Orchestrate | `scripts/run_pipeline.py` | Run ingest → simplify → extract → compare in one command |

Supporting modules: `scripts/prompt_templates.py` (prompt guardrails and JSON shape), `scripts/text_section_extract.py` (rule-based parsing of `simplified_text`).

Sample input: `data/raw_labels/sample_labels.json`.

---

## Requirements

- **Python 3.11+** (3.13 used in development is fine)
- Standard library for ingest. For **OpenAI** or **Groq** simplification, install dependencies (Groq uses the same OpenAI-compatible client):

  ```bash
  pip install -r requirements.txt
  # or: pip install openai
  ```

---

## Secrets and configuration

- **Never commit** API keys or paste them into issues/chat. Use a **local** `.env` file (keep it gitignored).
- Recommended variables:
  - **`OPENFDA_API_KEY`** — optional; higher openFDA rate limits (`scripts/ingest_data.py` loads `.env` from the repo root automatically).
  - **`OPENAI_API_KEY`** — for `--provider openai` on simplification.
  - **`OPENAI_MODEL`** — optional override (default `gpt-4o-mini`).
  - **`GROQ_API_KEY`** — for `--provider groq` (same SDK; base URL `https://api.groq.com/openai/v1`).
  - **`GROQ_SIMPLIFY_MODEL`** — optional Groq model id (default `llama-3.1-8b-instant`). Override with `--model`.
  - **`GROQ_OPENAI_BASE_URL`** — optional; only if Groq changes the compatible endpoint.
  - **`SIMPLIFY_MAX_LLM_LABEL_CHARS`** — optional cap on label characters sent to the LLM (default `8000`). Lower if Groq returns “request too large” on free tier; raise for OpenAI.

`scripts/simplify_labels.py` loads `.env` from the repo root (like ingest). Use `--max-llm-chars` to override per run.
- If a key was ever exposed, **revoke it** in the provider dashboard and rotate.

---

## 1. Data ingestion (openFDA)

Default behavior fetches a **curated list** of generic drugs. Other modes: single drug, bulk unfiltered pagination, or re-export from existing JSON.

```bash
# Curated list → data/drug_labels.json
python3 scripts/ingest_data.py --output drug_labels.json

# One generic name
python3 scripts/ingest_data.py --drug ibuprofen --output drug_labels.json

# Bulk (unfiltered), first N labels
python3 scripts/ingest_data.py --bulk 500 --output drug_labels.json
```

**CSV export** — use a `.csv` path or `--format csv`:

```bash
python3 scripts/ingest_data.py --output drug_labels.csv
python3 scripts/ingest_data.py --from-json data/drug_labels.json --output drug_labels.csv --format csv
```

`--from-json` performs export only (no API calls). If `OPENFDA_API_KEY` is set, the first request logs that authenticated (higher-limit) access is enabled; the key is never printed.

---

## 2. Label simplification

Prompt design enforces **safety preservation**: dosage, boxed warning, warnings, contraindications, and interactions must not be dropped, softened incorrectly, or fabricated. Output is structured JSON plus a combined `simplified_text`.

**Self-test** (no API key; uses sample labels):

```bash
python3 scripts/simplify_labels.py --self-test
```

**Local provider** (deterministic reformat; no API):

```bash
python3 scripts/simplify_labels.py \
  --provider local \
  --input data/raw_labels/sample_labels.json \
  --output simplified_labels.json \
  --pretty
```

**OpenAI** (install `openai`, set `OPENAI_API_KEY`):

```bash
python3 scripts/simplify_labels.py \
  --provider openai \
  --model gpt-4o-mini \
  --input data/raw_labels/sample_labels.json \
  --output simplified_labels.json \
  --pretty
```

**Groq** (install `openai`, set `GROQ_API_KEY`; uses OpenAI-compatible chat completions. Pick any model your Groq account supports, e.g. `llama-3.1-8b-instant`):

```bash
python3 scripts/simplify_labels.py \
  --provider groq \
  --model llama-3.1-8b-instant \
  --input data/raw_labels/sample_labels.json \
  --output simplified_labels.json \
  --pretty
```

Typical pipeline input is ingest output, e.g. `data/drug_labels.json`, as a JSON array of records.

---

## 3. Structured extraction

Produces aligned JSON for **original** (openFDA-shaped) or **simplified** records.

**From original labels:**

```bash
python3 scripts/extract_labels.py --source original --input data/drug_labels.json --output extracted_original.json
```

**From simplified labels** (structured JSON fields):

```bash
python3 scripts/extract_labels.py --source simplified --input simplified_labels.json --output extracted_simplified.json
```

**Simplified modes** (`--simplified-mode`):

- **`structured`** (default): read `dosage`, `warnings`, etc. from JSON fields.
- **`from_text`**: parse **only** `simplified_text` section headings.
- **`hybrid`**: structured first, fill gaps from `simplified_text`.

```bash
python3 scripts/extract_labels.py --source simplified --input simplified_labels.json \
  --output extracted_simplified_from_text.json --simplified-mode from_text
```

---

## 4. Compare extractions

Heuristic report (not clinical validation): e.g. dropped fields, possible dose token mismatches, long-prefix content loss.

```bash
python3 scripts/compare_extractions.py \
  --original extracted_original.json \
  --simplified extracted_simplified.json \
  --output comparison_report.json
```

---

## 5. One-command pipeline

From repo root — **ingest** one generic name, **simplify**, **extract**, **compare**:

```bash
python3 scripts/run_pipeline.py --drug ibuprofen --simplify-provider local
```

With OpenAI (set `OPENAI_API_KEY`):

```bash
python3 scripts/run_pipeline.py --drug ibuprofen --simplify-provider openai
```

With Groq (set `GROQ_API_KEY`; optional `--simplify-model ...`):

```bash
python3 scripts/run_pipeline.py --drug ibuprofen --simplify-provider groq --simplify-model llama-3.1-8b-instant
```

Skip re-fetch if `data/drug_labels.json` is already present:

```bash
python3 scripts/run_pipeline.py --skip-ingest --simplify-provider local
```

Also run text-only simplified extraction and a second report:

```bash
python3 scripts/run_pipeline.py --drug ibuprofen --simplify-provider local --extract-text
```

---

## Tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pytest openai   # openai only if you run OpenAI-backed tests/code paths
python3 -m pytest scripts/ -v
```

`conftest.py` adds `scripts/` to `sys.path` so tests can import pipeline modules.

---

## Generated artifacts

Large or local outputs (JSON/CSV from runs, logs, `outputs/`, etc.) are listed in **`.gitignore`**. Prefer committing **source** and **small fixtures** (e.g. `data/raw_labels/sample_labels.json`), not full fetched corpora or secrets.

---

## License / disclaimer

This project is for **research and coursework**. It is **not** medical advice. Do not use outputs as a substitute for professional labeling, prescribing, or clinical decision-making.
