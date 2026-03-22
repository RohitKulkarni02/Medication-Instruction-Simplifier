# Medication-Instruction-Simplifier

**CS6180 — Group 14**

Structured extraction of safety-critical fields from medication labels (original OpenFDA-style JSON vs. simplified JSON) into one unified schema. See **[PROJECT.md](PROJECT.md)** for full documentation: layout, CLI, schema, tests, and dependencies.

## Quick start

```bash
pip install -r requirements.txt
pytest test_extract_labels.py -v
```

```bash
# Place drug_labels.json or simplified_labels.json next to extract_labels.py, then:
python extract_labels.py --source original
python extract_labels.py --source simplified
```

Course proposal: `CS6180_Project_Proposal.pdf`.
