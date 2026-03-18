# Medication-Instruction-Simplifier
CS6180 - Group 14 

## Person 3: LLM Simplification Pipeline (Week 1-2)

### Files
- `scripts/prompt_templates.py`: prompt design (guardrails + JSON output schema)
- `scripts/simplify_labels.py`: pipeline to generate `simplified_text`
- `data/raw_labels/sample_labels.json`: small sample dataset for local testing

### Prompt guardrails (high level)
The prompt requires the model to preserve adherence- and safety-critical content:
dosage instructions, warnings, contraindications, and drug interactions.
It also asks for output as strict JSON (patient-friendly `simplified_text` plus structured safety fields).

### Run the local self-test
This uses the included sample labels and a local “reformat-only” backend (no API key needed):

```bash
python3 scripts/simplify_labels.py --self-test
```

### Run the simplification pipeline (local mode)
```bash
python3 scripts/simplify_labels.py \
  --provider local \
  --input data/raw_labels/sample_labels.json \
  --output outputs/simplified_labels/sample_simplified.json \
  --pretty
```

### Run with OpenAI (optional)
If you later install the `openai` package and set `OPENAI_API_KEY`, you can run:

```bash
python3 scripts/simplify_labels.py \
  --provider openai \
  --model gpt-4o-mini \
  --input data/raw_labels/sample_labels.json \
  --output outputs/simplified_labels/sample_simplified_openai.json \
  --pretty
```
