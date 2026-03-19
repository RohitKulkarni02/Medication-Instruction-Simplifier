import os
import requests
import json
import re
import time
import logging
import argparse
from datetime import datetime

API_KEY = "PZAZh10IvwqVKGd4GWngojnFbGcBNnz1DOWGsnXg"
BASE_URL = "https://api.fda.gov/drug/label.json"
CALL_DELAY = 0.5

# These are the top priority fields - used for label scoring and quick access to review
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

CURATED_DRUGS = [
    # OTC
    "ibuprofen",  # NSAID, GI/cardiovascular warnings often overlooked
    "acetaminophen",  # Liver toxicity risk at >4g/day, critical max dose
    "diphenhydramine",  # Sedation warnings, anticholinergic risks in elderly
    "loratadine",  # Low-risk baseline, useful as a control comparison
    "omeprazole",  # Long-term use warnings (bone loss, magnesium), often misused
    "cetirizine",  # Drowsiness warnings vary by individual, dosing nuances
    "aspirin",  # Blood thinning risk, Reye's syndrome warning in children
    "naproxen",  # Similar to ibuprofen but longer half-life, different dosing

    # Prescription
    "amoxicillin",  # Antibiotic, allergy/anaphylaxis warnings, resistance concerns
    "metformin",  # Boxed warning (lactic acidosis), renal contraindications
    "lisinopril",  # ACE inhibitor, pregnancy contraindication (category D), angioedema risk
    "atorvastatin",  # Liver/muscle toxicity warnings, many drug interactions
    "metoprolol",  # Beta-blocker, abrupt withdrawal risk, heart rate considerations
    "sertraline",  # SSRI, boxed warning (suicidality in young adults), serotonin syndrome
    "prednisone",  # Corticosteroid, complex taper instructions, immunosuppression risk
    "azithromycin",  # QT prolongation risk, cardiac warnings

    # High-risk / Controlled Substances
    "methotrexate",  # Boxed warning (fatal toxicity), weekly-not-daily dosing confusion kills patients
    "tofacitinib",  # JAK inhibitor, boxed warning (serious infections, malignancy, thrombosis)
    "warfarin",  # Boxed warning (bleeding), extremely narrow therapeutic window, massive interaction list
    "isotretinoin", # Boxed warning for severe birth defects. Requires a pledge and pregnancy testing
    "clozapine", # Boxed warning for severe neutropenia (fatal infections), myocarditis, and seizures.
    "oxycodone",  # Schedule II opioid, boxed warning (addiction, respiratory depression, death)
    "alprazolam",  # Schedule IV benzo, boxed warning (opioid combo, dependence, withdrawal)
    "amphetamine",  # Schedule II stimulant (Adderall), cardiovascular/psychiatric warnings
    "diazepam",  # Schedule IV benzo, dependence/withdrawal, opioid interaction warnings
]

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Text Cleaning
def text_clean(raw_text):
    """
    Cleans raw label text by stripped tags, removing section numbers/titles,
    normalizing text/spacing, and stripping any leading/trailing items
    """

    if not raw_text:
        return None

    # Remove HTML Tags
    text = re.sub(r"<[^>]+>", " ", raw_text)

    # Remove section numbers and prefixes
    text = re.sub(r"^\d+(\.\d+)?\s+[A-Z\s/&]+\n?", "", text.strip())

    # Remove subsection numbers
    text = re.sub(r"(\n\s*)\d+\.\d+\s+", r"\1", text)

    # HTML code conversion
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&#x2265;", "≥")
    text = text.replace("&#x2264;", "≤")
    text = text.replace("&#xB0;", "°")
    text = text.replace("&#x2019;", "'")
    text = text.replace("&#x201C;", '"')
    text = text.replace("&#x201D;", '"')

    # Whitespace normalisation
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text if text else None


# Extraction of Fields
def extract_all_fields(label):
    """
    Extracts and cleans all text fields from the label
    """

    all_fields = {}

    for key, value in label.items():
        if key in SKIP_FIELDS:
            continue
        if key.endswith("_table"): # We are ignoring these fields as they create an HTML table.
            continue

        if isinstance(value, list) and value and isinstance(value[0], str):
            cleaned = text_clean(" ".join(value))
            all_fields[key] = cleaned

    return all_fields

def build_full_label(all_fields):
    """
    Reconstructs the full label from the extracted text fields
    """

    fields = []
    for value in all_fields.values():
        if value:
            fields.append(value)

    full = "\n\n".join(fields)
    return full if full else None

# RECORD BUILDING

# Label Selection -
# When the API returns multiple labels for one drug (e.g. 83 results for alprazolam),
# this section picks the best one — scored by most safety-critical fields present + most recent effective date.

def score_label(label):
    """
    Scores a label by number of safety-critical fields present and the effective time published.
    Higher score the better. It'll select the more complete and most recent label.
    """

    field_count = 0
    for f in SAFETY_CRITICAL_FIELDS:
        if label.get(f):
            field_count = field_count+1

    effective = label.get("effective_time", "18000501")
    try:
        date = datetime.strptime(effective, "%Y%m%d")
    except ValueError:
        date = datetime(1800, 5, 1)

    return field_count, date

def select_best_label(results):
    """
    Picking the most complete and most recent label.
    """

    if not results:
        return None

    return max(results, key=score_label)


# API Fetch
def get_labels(drug_name, limit=20):
    """
    Fetches the drug labels for a drug by generic name. Pulls up to 20 labels to compare and pick from.
    Returns the best label record or None if there is nothing.
    """

    params = {
        "api_key": API_KEY,
        "search": f'openfda.generic_name:"{drug_name}"',
        "limit": limit,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            logger.warning(f"No results found for: {drug_name}")
            return None

        best = select_best_label(results)
        logger.info(
            f"Found {len(results)} labels for '{drug_name}', "
            f"selected label with {score_label(best)[0]} safety-critical fields, "
            f"effective date: {best.get('effective_time', 'unknown')}"
        )
        return best

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error for '{drug_name}': {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for '{drug_name}': {e}")
        return None
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Parse error for '{drug_name}': {e}")
        return None

def get_bulk_labels(limit=100, skip=0):
    """
    Fetches labels in bulk for building the large-scale dataset.
    100 is the maximum API request allowed by openFDA
    """
    params = {
        "api_key": API_KEY,
        "limit": min(limit, 100),
        "skip": skip,
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Bulk fetch error (skip={skip}): {e}")
        return []

#Record Building

def build_record(label):
    """
    Creates a clean, structured record ready for use from the raw openFDA data
    Structure:
    - Top level metadata such as drug name, brand name
    - Safety-critical fields pulled to the top for quick reading access
    - all_fields contains every extracted text field
    - full_label with all text concatenated
       - This can be fed to the LLM to get simplification for full label information
    """

    openfda = label.get("openfda", {})

    drug_name = openfda.get("generic_name", [None])[0]
    if not drug_name:
        logger.warning(f"Skipping label with no drug name (spl_id: {label.get('id', 'unknown')})")
        return None

    brand_name = openfda.get("brand_name", [None])[0]
    product_type = openfda.get("product_type", [None])[0]

    all_fields = extract_all_fields(label)

    record = {
        # Metadata
        "drug_name": drug_name,
        "brand_name": brand_name,
        "product_type": product_type,
        "effective_date": label.get("effective_time"),
        "spl_id": label.get("id"),

        #Safety-critical fields
        "dosage_and_administration": all_fields.get("dosage_and_administration"),
        "warnings": all_fields.get("warnings") or all_fields.get("warnings_and_cautions"),
        "contraindications": all_fields.get("contraindications"),
        "drug_interactions": all_fields.get("drug_interactions"),
        "boxed_warning": all_fields.get("boxed_warning"),
        "indications_and_usage": all_fields.get("indications_and_usage"),
        "adverse_reactions": all_fields.get("adverse_reactions"),

        # All extracted fields
        "all_fields": all_fields,

        #Full concatenated label text
        "full_label": build_full_label(all_fields),
    }

    # Logging the missing safety-critical fields. Excludes black-box warning as this is not a common item for labels.
    safety_fields_in_records = [
        "dosage_and_administration", "warnings", "contraindications",
        "drug_interactions", "indications_and_usage", "adverse_reactions",
    ]

    missing = []
    for k in safety_fields_in_records:
        value = record.get(k)
        if value is None:
            missing.append(k)

    if missing:
        logger.warning(f"Missing safety-critical fields for '{drug_name or 'UNKNOWN'}': {', '.join(missing)}")


    return record

# Pipeline

def run_set_drug_list(drug_list=None):
    """
    Gets and processes the labels for the curated drug list above.
    """

    drugs = drug_list or CURATED_DRUGS
    dataset = []

    logger.info(f"Starting curated fetch for {len(drugs)} drugs...")

    for i, drug in enumerate(drugs):
        logger.info(f"[{i+1}/{len(drugs)}] Fetching: {drug}")
        label = get_labels(drug)

        if label:
            record = build_record(label)
            dataset.append(record)
        else:
            logger.warning(f"Skipping '{drug}' — no data returned")

        time.sleep(CALL_DELAY)

    logger.info(f"Curated fetch complete. {len(dataset)}/{len(drugs)} drugs retrieved.")

    return dataset

def run_bulk(total=1000):
    """
    Gets labels in bulk with pagination to work around API limits.
    The total above is for when a total is not specified by the user.
    """

    dataset = []
    skip = 0
    per_page = 100

    logger.info(f"Starting bulk fetch for {total} labels...")

    while len(dataset) < total:
        labels = get_bulk_labels(limit=per_page, skip=skip)

        if not labels:
            logger.warning(f"No more results at skip={skip}. Stopping.")
            break

        for label in labels:
            record = build_record(label)
            if record:
                dataset.append(record)

        skip += per_page
        logger.info(f"Fetched {len(dataset)} records so far...")
        time.sleep(CALL_DELAY)

    dataset = dataset[:total]
    logger.info(f"Bulk fetch complete. {len(dataset)} records retrieved.")
    return dataset

def run_single_drug(drug_name):
    """
    Gets and processes label for single specified drug.
    """

    logger.info(f"Fetching single drug: {drug_name}")
    label = get_labels(drug_name)

    if label:
        record = build_record(label)
        return [record]
    else:
        logger.warning(f"No data found for '{drug_name}'")
        return []

def export_dataset(dataset, file_name="drug_labels.json"):
    """
    Exports to JSON
    """

    out_path = f"data/{file_name}"

    os.makedirs("data", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"Dataset exported to {out_path} ({len(dataset)} records)")

def print_summary(dataset):
    """
    REMOVE DELETE PRIOR TO SUBMISSION - USED FOR TESTING/QA
    """

    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(dataset)}")

    # Count by product type
    types = {}
    for r in dataset:
        t = r.get("product_type") or "UNKNOWN"
        types[t] = types.get(t, 0) + 1
    print(f"\nBy product type:")
    for t, count in sorted(types.items()):
        print(f"  {t}: {count}")

    # Safety-critical field coverage
    safety_fields = [
        "dosage_and_administration", "warnings", "contraindications",
        "drug_interactions", "boxed_warning", "indications_and_usage",
        "adverse_reactions",
    ]
    print(f"\nSafety-critical field coverage:")
    for field in safety_fields:
        present = sum(1 for r in dataset if r.get(field) is not None)
        pct = (present / len(dataset) * 100) if dataset else 0
        print(f"  {field}: {present}/{len(dataset)} ({pct:.0f}%)")

    # All fields overview
    if dataset:
        all_field_names = set()
        for r in dataset:
            all_field_names.update(r.get("all_fields", {}).keys())
        print(f"\nTotal unique fields extracted across all records: {len(all_field_names)}")

        # Show coverage for every field found
        print(f"\nAll field coverage:")
        for field in sorted(all_field_names):
            present = sum(
                1 for r in dataset
                if r.get("all_fields", {}).get(field) is not None
            )
            pct = (present / len(dataset) * 100) if dataset else 0
            print(f"  {field}: {present}/{len(dataset)} ({pct:.0f}%)")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Drug Label Data Pipeline Parser"
    )
    parser.add_argument(
        "--drug", type=str, help="Fetch a single drug by generic name"
    )
    parser.add_argument(
        "--bulk", type=int, help="Fetch N labels in bulk (no filter)"
    )
    parser.add_argument(
        "--output", type=str, default="drug_labels.json",
        help="Output filename (default: drug_labels.json)"
    )

    args = parser.parse_args()

    if args.drug:
        dataset = run_single_drug(args.drug)
    elif args.bulk:
        dataset = run_bulk(total=args.bulk)
    else:
        dataset = run_set_drug_list()

    if dataset:
        export_dataset(dataset, file_name=args.output)
        print_summary(dataset)
    else:
        logger.error("No data collected. Check logs for errors.")











































