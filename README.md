# Drug Label Dataset Builder

This project pulls drug labeling data from the openFDA Drug Label API and exports it into a clean JSON dataset for downstream analysis or LLM use.

## What it does

The script:

- fetches drug label data from the openFDA `drug/label` endpoint
- supports three modes:
  - a curated list of drugs
  - one specific drug
  - bulk label collection
- cleans and normalizes text fields
- extracts safety-critical sections for easier review
- builds a full concatenated label text field for each record
- exports the results to JSON
- logs pipeline activity to `pipeline.log`

## Main file

- `build_dataset.py` — main pipeline script

## Output structure

Each exported record includes:

- `drug_name`
- `brand_name`
- `product_type`
- `effective_date`
- `spl_id`
- safety-critical fields such as:
  - `dosage_and_administration`
  - `warnings`
  - `contraindications`
  - `drug_interactions`
  - `boxed_warning`
  - `indications_and_usage`
  - `adverse_reactions`
- `all_fields` — every extracted text field from the label
- `full_label` — all extracted text joined together into one field


## How to run

### 1. Run the curated drug list

```bash
python build_dataset.py
```

This uses the built-in curated set of drugs defined in `CURATED_DRUGS`.

### 2. Run a single drug

```bash
python build_dataset.py --drug ibuprofen
```

### 3. Run bulk collection

```bash
python build_dataset.py --bulk 50
```

This fetches up to 50 label records in bulk.

### 4. Choose a custom output filename

```bash
python build_dataset.py --drug ibuprofen --output ibuprofen.json
```

## Output files

By default, output is written to:

```bash
data/drug_labels.json
```

The script also creates:

```bash
pipeline.log
```

## How label selection works

When multiple label results are returned for a drug, the script selects the “best” label by prioritizing:

1. the number of safety-critical fields present
2. the most recent `effective_time`

## Data cleaning behavior

The script cleans extracted text by:

- removing HTML tags
- converting common HTML entities
- removing some section numbering/prefixes
- normalizing whitespace
- skipping metadata-heavy or display-only fields
- skipping fields ending in `_table`

## Notes

- Bulk mode uses pagination with a maximum of 100 records per request.
- The script adds a short delay between requests to avoid hammering the API.
- Records without a usable generic drug name are skipped.
- Missing safety-critical fields are logged as warnings.
