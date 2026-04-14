from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from datetime import datetime
import urllib.parse
import urllib.request

BASE_URL = "https://api.fda.gov/drug/label.json"
CALL_DELAY = 0.5

SAFETY_CRITICAL_FIELDS = [
    "dosage_and_administration",
    "warnings",
    "warnings_and_cautions",
    "contraindications",
    "drug_interactions",
    "boxed_warning",
    "indications_and_usage",
    "adverse_reactions",
]

SKIP_FIELDS = [
    "id",
    "set_id",
    "version",
    "effective_time",
    "openfda",
    "spl_product_data_elements",
    "package_label_principal_display_panel",
]

CSV_COLUMNS = [
    "drug_name",
    "brand_name",
    "product_type",
    "effective_date",
    "spl_id",
    "dosage_and_administration",
    "warnings",
    "contraindications",
    "drug_interactions",
    "boxed_warning",
    "indications_and_usage",
    "adverse_reactions",
    "full_label",
    "all_fields_json",
]

CURATED_DRUGS = [
    "ibuprofen",
    "acetaminophen",
    "diphenhydramine",
    "loratadine",
    "omeprazole",
    "cetirizine",
    "aspirin",
    "naproxen",
    "amoxicillin",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "metoprolol",
    "sertraline",
    "prednisone",
    "azithromycin",
    "methotrexate",
    "tofacitinib",
    "warfarin",
    "isotretinoin",
    "clozapine",
    "oxycodone",
    "alprazolam",
    "amphetamine",
    "diazepam",
]

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PIPELINE_LOG = os.path.join(_ROOT, "outputs", "pipeline.log")
os.makedirs(os.path.dirname(_PIPELINE_LOG), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(_PIPELINE_LOG), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_dotenv_loaded = False
_openfda_key_logged = False


def _load_dotenv() -> None:
    """Lightweight .env loader to avoid extra dependencies."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                os.environ[key] = value


def _ensure_dotenv_loaded() -> None:
    """Load repo-root .env once before openFDA calls (CLI or imported usage)."""
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    _load_dotenv()
    _dotenv_loaded = True


def text_clean(raw_text: str | None) -> str | None:
    if not raw_text:
        return None

    text = re.sub(r"<[^>]+>", " ", raw_text)
    text = re.sub(r"^\d+(\.\d+)?\s+[A-Z\s/&]+\n?", "", text.strip())
    text = re.sub(r"(\n\s*)\d+\.\d+\s+", r"\1", text)

    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#x2265;", ">=")
    text = text.replace("&#x2264;", "<=")
    text = text.replace("&#xB0;", " degrees ")
    text = text.replace("&#x2019;", "'")
    text = text.replace("&#x201C;", '"')
    text = text.replace("&#x201D;", '"')

    text = re.sub(r"\s+", " ", text).strip()
    return text if text else None


def extract_all_fields(label: dict) -> dict[str, str | None]:
    all_fields: dict[str, str | None] = {}

    for key, value in label.items():
        if key in SKIP_FIELDS or key.endswith("_table"):
            continue
        if isinstance(value, list) and value and isinstance(value[0], str):
            all_fields[key] = text_clean(" ".join(value))

    return all_fields


def build_full_label(all_fields: dict[str, str | None]) -> str | None:
    fields = [v for v in all_fields.values() if v]
    full = "\n\n".join(fields)
    return full if full else None


def score_label(label: dict) -> tuple[int, datetime]:
    field_count = 0
    for field in SAFETY_CRITICAL_FIELDS:
        if label.get(field):
            field_count += 1

    effective = label.get("effective_time", "18000501")
    try:
        date = datetime.strptime(effective, "%Y%m%d")
    except ValueError:
        date = datetime(1800, 5, 1)

    return field_count, date


def select_best_label(results: list[dict]) -> dict | None:
    if not results:
        return None
    return max(results, key=score_label)


def _auth_params() -> dict[str, str]:
    global _openfda_key_logged
    _ensure_dotenv_loaded()
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    if api_key:
        if not _openfda_key_logged:
            logger.info("openFDA requests include api_key (OPENFDA_API_KEY set; higher rate limits)")
            _openfda_key_logged = True
        return {"api_key": api_key}
    if not _openfda_key_logged:
        logger.debug(
            "OPENFDA_API_KEY not set; openFDA allows unauthenticated requests with lower rate limits"
        )
        _openfda_key_logged = True
    return {}


def _http_get_json(params: dict[str, str | int], timeout: int = 30) -> dict:
    query = urllib.parse.urlencode(params, doseq=True)
    url = f"{BASE_URL}?{query}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def get_labels(drug_name: str, limit: int = 20) -> dict | None:
    params = {
        **_auth_params(),
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": limit,
    }
    try:
        data = _http_get_json(params=params, timeout=30)
        results = data.get("results", [])
        if not results:
            logger.warning("No results found for: %s", drug_name)
            return None

        best = select_best_label(results)
        logger.info(
            "Found %s labels for '%s'; selected with %s safety fields, effective=%s",
            len(results),
            drug_name,
            score_label(best)[0] if best else 0,
            best.get("effective_time", "unknown") if best else "unknown",
        )
        return best
    except urllib.error.HTTPError as e:
        logger.error("HTTP error for '%s': %s", drug_name, e)
        return None
    except urllib.error.URLError as e:
        logger.error("Request error for '%s': %s", drug_name, e)
        return None
    except (KeyError, json.JSONDecodeError) as e:
        logger.error("Parse error for '%s': %s", drug_name, e)
        return None


def get_bulk_labels(limit: int = 100, skip: int = 0) -> list[dict]:
    params = {**_auth_params(), "limit": min(limit, 100), "skip": skip}
    try:
        return _http_get_json(params=params, timeout=30).get("results", [])
    except Exception as e:  # noqa: BLE001
        logger.error("Bulk fetch error (skip=%s): %s", skip, e)
        return []


def build_record(label: dict) -> dict | None:
    openfda = label.get("openfda", {})
    drug_name = openfda.get("generic_name", [None])[0]
    if not drug_name:
        logger.warning("Skipping label with no drug name (spl_id: %s)", label.get("id", "unknown"))
        return None

    brand_name = openfda.get("brand_name", [None])[0]
    product_type = openfda.get("product_type", [None])[0]
    all_fields = extract_all_fields(label)

    record = {
        "drug_name": drug_name,
        "brand_name": brand_name,
        "product_type": product_type,
        "effective_date": label.get("effective_time"),
        "spl_id": label.get("id"),
        "dosage_and_administration": all_fields.get("dosage_and_administration"),
        "warnings": all_fields.get("warnings") or all_fields.get("warnings_and_cautions"),
        "contraindications": all_fields.get("contraindications"),
        "drug_interactions": all_fields.get("drug_interactions"),
        "boxed_warning": all_fields.get("boxed_warning"),
        "indications_and_usage": all_fields.get("indications_and_usage"),
        "adverse_reactions": all_fields.get("adverse_reactions"),
        "all_fields": all_fields,
        "full_label": build_full_label(all_fields),
    }

    safety_fields = [
        "dosage_and_administration",
        "warnings",
        "contraindications",
        "drug_interactions",
        "indications_and_usage",
        "adverse_reactions",
    ]
    missing = [k for k in safety_fields if record.get(k) is None]
    if missing:
        logger.warning("Missing safety fields for '%s': %s", drug_name, ", ".join(missing))

    return record


def run_set_drug_list(drug_list: list[str] | None = None) -> list[dict]:
    drugs = drug_list or CURATED_DRUGS
    dataset: list[dict] = []
    logger.info("Starting curated fetch for %s drugs", len(drugs))
    for i, drug in enumerate(drugs):
        logger.info("[%s/%s] Fetching: %s", i + 1, len(drugs), drug)
        label = get_labels(drug)
        if label:
            record = build_record(label)
            if record:
                dataset.append(record)
        else:
            logger.warning("Skipping '%s' — no data returned", drug)
        time.sleep(CALL_DELAY)

    logger.info("Curated fetch complete. %s/%s drugs retrieved.", len(dataset), len(drugs))
    return dataset


def run_bulk(total: int = 1000) -> list[dict]:
    dataset: list[dict] = []
    skip = 0
    per_page = 100
    logger.info("Starting bulk fetch for %s labels", total)

    while len(dataset) < total:
        labels = get_bulk_labels(limit=per_page, skip=skip)
        if not labels:
            logger.warning("No more results at skip=%s. Stopping.", skip)
            break
        for label in labels:
            record = build_record(label)
            if record:
                dataset.append(record)
        skip += per_page
        logger.info("Fetched %s records so far...", len(dataset))
        time.sleep(CALL_DELAY)

    return dataset[:total]


def run_single_drug(drug_name: str) -> list[dict]:
    logger.info("Fetching single drug: %s", drug_name)
    label = get_labels(drug_name)
    if label:
        record = build_record(label)
        return [record] if record else []
    logger.warning("No data found for '%s'", drug_name)
    return []


def _resolve_output_path(output: str) -> str:
    if os.path.isabs(output):
        return output
    if output.startswith("data/"):
        return output
    return os.path.join("data", output)


def _export_dataset_json(dataset: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def _record_to_csv_row(record: dict) -> dict[str, str]:
    row: dict[str, str] = {}
    for key in CSV_COLUMNS:
        if key == "all_fields_json":
            af = record.get("all_fields")
            row[key] = json.dumps(af, ensure_ascii=False) if af is not None else ""
            continue
        val = record.get(key)
        row[key] = "" if val is None else str(val)
    return row


def _export_dataset_csv(dataset: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for rec in dataset:
            writer.writerow(_record_to_csv_row(rec))


def export_dataset(dataset: list[dict], output: str, *, fmt: str = "json") -> None:
    out_path = _resolve_output_path(output)
    resolved_fmt = fmt
    if resolved_fmt == "auto":
        resolved_fmt = "csv" if out_path.lower().endswith(".csv") else "json"

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    if resolved_fmt == "csv":
        _export_dataset_csv(dataset, out_path)
    else:
        _export_dataset_json(dataset, out_path)
    logger.info("Dataset exported to %s (%s records, %s)", out_path, len(dataset), resolved_fmt)


def load_dataset_json(path: str) -> list[dict]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [path]
    if not os.path.isabs(path):
        candidates.extend(
            [
                os.path.join(root, path),
                os.path.join(root, "data", path),
                _resolve_output_path(path),
            ]
        )
    for p in candidates:
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array in {p}")
            return data
    raise FileNotFoundError(f"Could not find JSON file: {path}")


def print_summary(dataset: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total records: {len(dataset)}")

    types: dict[str, int] = {}
    for record in dataset:
        product_type = record.get("product_type") or "UNKNOWN"
        types[product_type] = types.get(product_type, 0) + 1
    print("\nBy product type:")
    for product_type, count in sorted(types.items()):
        print(f"  {product_type}: {count}")

    safety_fields = [
        "dosage_and_administration",
        "warnings",
        "contraindications",
        "drug_interactions",
        "boxed_warning",
        "indications_and_usage",
        "adverse_reactions",
    ]
    print("\nSafety-critical field coverage:")
    for field in safety_fields:
        present = sum(1 for record in dataset if record.get(field) is not None)
        pct = (present / len(dataset) * 100) if dataset else 0
        print(f"  {field}: {present}/{len(dataset)} ({pct:.0f}%)")


def main() -> int:
    _ensure_dotenv_loaded()
    parser = argparse.ArgumentParser(description="openFDA drug label ingestion (JSON or CSV export)")
    parser.add_argument("--drug", type=str, help="Fetch a single drug by generic name")
    parser.add_argument("--bulk", type=int, help="Fetch N labels in bulk (no filter)")
    parser.add_argument(
        "--from-json",
        type=str,
        metavar="PATH",
        help="Load an existing ingest JSON array and export only (no API calls)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drug_labels.json",
        help="Output filename/path (default: data/drug_labels.json)",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "csv"),
        default="auto",
        help="Export format (default: infer from .csv/.json extension)",
    )
    args = parser.parse_args()

    if args.from_json:
        try:
            dataset = load_dataset_json(args.from_json)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.error("Could not read %s: %s", args.from_json, e)
            return 1
        export_dataset(dataset, output=args.output, fmt=args.format)
        print_summary(dataset)
        return 0

    if args.drug:
        dataset = run_single_drug(args.drug)
    elif args.bulk:
        dataset = run_bulk(total=args.bulk)
    else:
        dataset = run_set_drug_list()

    if dataset:
        export_dataset(dataset, output=args.output, fmt=args.format)
        print_summary(dataset)
        return 0

    logger.error("No data collected. Check logs for errors.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

