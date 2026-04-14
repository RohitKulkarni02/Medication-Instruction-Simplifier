"""
LLM-as-judge: compare extracted original vs simplified safety fields (semantic).

Reads the same JSON shape as compare_extractions.py; writes `outputs/evaluation_report.json` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any

from judge_taxonomy import (
    DROPPED,
    JUDGE_CATEGORY_DESCRIPTIONS,
    PRESERVED,
    SOFTENED,
)

# Fourth outcome when the model output cannot be parsed after retry.
ERROR = "ERROR"

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"

SAFETY_FIELDS = [
    "boxed_warning",
    "dosage",
    "warnings",
    "contraindications",
    "interactions",
]

DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"
SNIPPET_LEN = 200
DEFAULT_MAX_OUTPUT_TOKENS = 768
RAW_TRUNCATE = 1200


def _load_dotenv() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(root, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                os.environ[key] = value


def field_to_text(value: Any) -> str:
    """Normalize extracted field values (str or list) to a single string."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(parts)
    s = str(value).strip()
    return s


def snippet(text: str, max_len: int = SNIPPET_LEN) -> str:
    t = text or ""
    return t if len(t) <= max_len else t[:max_len]


def build_judge_system_prompt() -> str:
    return (
        "You are a clinical labeling safety evaluator. "
        "You compare ONE field from an FDA-style medication label (original) "
        "to the same field from a patient-simplified version. "
        "Respond ONLY with a single JSON object, no markdown fences, no preamble. "
        'Use exactly these keys: "judgment" and "explanation". '
        f'judgment must be one of: "{PRESERVED}", "{SOFTENED}", "{DROPPED}".'
    )


def build_judge_user_prompt(field: str, original_text: str, simplified_text: str) -> str:
    defs = "\n".join(
        f"- {k}: {JUDGE_CATEGORY_DESCRIPTIONS.get(k, '')}"
        for k in (PRESERVED, SOFTENED, DROPPED)
    )
    return (
        f"Field name: {field}\n\n"
        "Definitions:\n"
        f"{defs}\n\n"
        "Additional rules:\n"
        f'- "{PRESERVED}": all safety-critical information from the original is present '
        "in the simplified text, even if reworded or shortened for readability.\n"
        f'- "{SOFTENED}": core safety information is still present but urgency, specificity, '
        "or scope is weakened (e.g. 'contraindicated' softened to 'not recommended', "
        "specific drug names or numeric thresholds removed or vague).\n"
        f'- "{DROPPED}": safety-critical information from the original is missing, '
        "contradicted, or replaced with filler in the simplified text.\n\n"
        "Original text (full):\n"
        f"{original_text}\n\n"
        "Simplified text (full):\n"
        f"{simplified_text}\n\n"
        'Return JSON: {"judgment": "<PRESERVED|SOFTENED|DROPPED>", '
        '"explanation": "<1-2 sentences; if PRESERVED may be brief>"}'
    )


def _extract_first_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Empty model output.")
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return json.loads(candidate)
    raise ValueError("Could not extract a complete JSON object.")


def parse_judge_json(raw: str) -> dict[str, Any]:
    """Parse model output to a dict; may raise."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty content")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _extract_first_json_object(raw)
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object")
    return data


def normalize_judgment_dict(data: dict[str, Any]) -> dict[str, str]:
    j = data.get("judgment")
    if j is None:
        raise ValueError("missing judgment")
    judgment = str(j).strip().upper()
    if judgment not in (PRESERVED, SOFTENED, DROPPED):
        raise ValueError(f"invalid judgment: {j!r}")
    expl = data.get("explanation", "")
    explanation = str(expl).strip() if expl is not None else ""
    if judgment == PRESERVED and not explanation:
        explanation = "No material loss of safety-critical meaning detected."
    return {"judgment": judgment, "explanation": explanation}


def invoke_judge_completion_json_then_plain(
    client: Any,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    system: str,
    user: str,
) -> str:
    """Try chat completion with json_object; on API failure fall back to plain messages."""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    return (response.choices[0].message.content or "").strip()


def invoke_judge_completion_plain(client: Any, *, model: str, temperature: float, max_output_tokens: int, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def call_judge_llm(
    client: Any,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    system: str,
    user: str,
) -> tuple[dict[str, str], str]:
    """
    First model response (json_object with plain fallback inside first HTTP round-trip pair).
    If JSON parse or validation fails, one additional plain completion; then parse again.
    Returns (normalized dict, raw string from the successful parse attempt).
    """
    last_raw = invoke_judge_completion_json_then_plain(
        client,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system=system,
        user=user,
    )
    try:
        return normalize_judgment_dict(parse_judge_json(last_raw)), last_raw
    except Exception:
        last_raw = invoke_judge_completion_plain(
            client,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system=system,
            user=user,
        )
        return normalize_judgment_dict(parse_judge_json(last_raw)), last_raw


def recompute_summary(
    per_drug: list[dict[str, Any]],
    *,
    skipped_both_empty: int,
    skipped_no_original: int,
) -> dict[str, Any]:
    preserved = softened = dropped = errors = 0
    total = 0
    unmatched = 0
    for drug_row in per_drug:
        if drug_row.get("unmatched"):
            unmatched += 1
        for j in drug_row.get("judgments") or []:
            total += 1
            v = j.get("judgment")
            if v == PRESERVED:
                preserved += 1
            elif v == SOFTENED:
                softened += 1
            elif v == DROPPED:
                dropped += 1
            elif v == ERROR:
                errors += 1
    return {
        "total_judgments": total,
        "preserved": preserved,
        "softened": softened,
        "dropped": dropped,
        "errors": errors,
        "unmatched_drugs": unmatched,
        "skipped_both_empty": skipped_both_empty,
        "skipped_no_original": skipped_no_original,
    }


def write_report_atomic(path: str, report: dict[str, Any]) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="eval_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def evaluate_drug_pair(
    orig: dict[str, Any],
    simp: dict[str, Any] | None,
    client: Any,
    *,
    model_used: str,
    system: str,
    temperature: float,
    max_output_tokens: int,
    delay: float,
    call_llm: Any = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """
    Returns (judgments, skipped_both_empty_delta, skipped_no_original_delta).
    If simp is None, returns ([], 0, 0) — caller should record unmatched.
    """
    llm = call_llm or call_judge_llm
    judgments: list[dict[str, Any]] = []
    skipped_both = 0
    skipped_no_orig = 0
    if simp is None:
        return judgments, skipped_both, skipped_no_orig

    for field in SAFETY_FIELDS:
        o_txt = field_to_text(orig.get(field))
        s_txt = field_to_text(simp.get(field))

        if not o_txt and not s_txt:
            skipped_both += 1
            continue
        if not o_txt:
            skipped_no_orig += 1
            continue
        if not s_txt:
            judgments.append(
                {
                    "field": field,
                    "judgment": DROPPED,
                    "explanation": "Simplified field is empty while original has content.",
                    "original_snippet": snippet(o_txt),
                    "simplified_snippet": snippet(s_txt),
                }
            )
            continue

        user = build_judge_user_prompt(field, o_txt, s_txt)
        try:
            norm, _raw = llm(
                client,
                model=model_used,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system=system,
                user=user,
            )
            judgments.append(
                {
                    "field": field,
                    "judgment": norm["judgment"],
                    "explanation": norm["explanation"],
                    "original_snippet": snippet(o_txt),
                    "simplified_snippet": snippet(s_txt),
                }
            )
        except Exception as e:
            judgments.append(
                {
                    "field": field,
                    "judgment": ERROR,
                    "explanation": f"LLM response malformed after retry. {e!s}"[:RAW_TRUNCATE],
                    "original_snippet": snippet(o_txt),
                    "simplified_snippet": snippet(s_txt),
                }
            )

        if delay > 0:
            time.sleep(delay)

    return judgments, skipped_both, skipped_no_orig


def run_dry_run(
    original_list: list[dict[str, Any]],
    simplified_by_drug: dict[str, dict[str, Any]],
) -> None:
    system = build_judge_system_prompt()
    for orig in original_list:
        drug = orig.get("drug_name", "UNKNOWN")
        key = str(drug).upper()
        simp = simplified_by_drug.get(key)
        if not simp:
            continue
        for field in SAFETY_FIELDS:
            o_txt = field_to_text(orig.get(field))
            s_txt = field_to_text(simp.get(field))
            if o_txt and s_txt:
                print("--- DRY RUN (first LLM-evaluable field) ---")
                print("SYSTEM:\n", system)
                print("USER:\n", build_judge_user_prompt(field, o_txt, s_txt))
                return
    print("Dry run: no LLM-evaluable field pairs found (need both original and simplified non-empty).", file=sys.stderr)


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="LLM-as-judge over extracted original vs simplified labels.")
    parser.add_argument("--original", default="outputs/extracted_original.json", help="Path to extracted_original.json")
    parser.add_argument("--simplified", default="outputs/extracted_simplified.json", help="Path to extracted_simplified.json")
    parser.add_argument("--output", default="outputs/evaluation_report.json", help="Write evaluation_report.json here")
    parser.add_argument("--provider", choices=["openai", "groq"], required=True)
    parser.add_argument("--model", default="", help="Model id (defaults: gpt-4o / llama-3.1-70b-versatile)")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep after each LLM API call")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--dry-run", action="store_true", help="Print prompts for first evaluable field; no API")
    args = parser.parse_args()

    with open(args.original, encoding="utf-8") as f:
        original_list = json.load(f)
    with open(args.simplified, encoding="utf-8") as f:
        simplified_list = json.load(f)
    if not isinstance(original_list, list) or not isinstance(simplified_list, list):
        print("Error: expected JSON arrays for --original and --simplified.", file=sys.stderr)
        return 2

    simplified_by_drug: dict[str, dict[str, Any]] = {}
    for row in simplified_list:
        dn = row.get("drug_name")
        if dn:
            simplified_by_drug[str(dn).upper()] = row

    if args.dry_run:
        run_dry_run(original_list, simplified_by_drug)
        return 0

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        print("Error: openai package is required. pip install openai", file=sys.stderr)
        raise SystemExit(2) from e

    if args.provider == "openai":
        if not os.environ.get("OPENAI_API_KEY", "").strip():
            print("Error: OPENAI_API_KEY is not set.", file=sys.stderr)
            return 2
        model_used = args.model.strip() or DEFAULT_OPENAI_MODEL
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").strip())
    else:
        if not os.environ.get("GROQ_API_KEY", "").strip():
            print("Error: GROQ_API_KEY is not set.", file=sys.stderr)
            return 2
        model_used = (
            args.model.strip()
            or os.environ.get("GROQ_JUDGE_MODEL", "").strip()
            or DEFAULT_GROQ_MODEL
        )
        base = os.environ.get("GROQ_OPENAI_BASE_URL", GROQ_OPENAI_BASE_URL).strip() or GROQ_OPENAI_BASE_URL
        client = OpenAI(api_key=os.environ.get("GROQ_API_KEY", "").strip(), base_url=base)

    system = build_judge_system_prompt()
    per_drug: list[dict[str, Any]] = []
    skipped_both_empty = 0
    skipped_no_original = 0

    for orig in original_list:
        drug = orig.get("drug_name", "UNKNOWN")
        key = str(drug).upper()
        simp = simplified_by_drug.get(key)

        if not simp:
            print(f"WARNING: no simplified record for drug_name={drug!r}", file=sys.stderr)
            per_drug.append({"drug_name": drug, "judgments": [], "unmatched": True})
            write_report_atomic(
                args.output,
                {
                    "schema_version": "1",
                    "model_used": model_used,
                    "summary": recompute_summary(per_drug, skipped_both_empty=skipped_both_empty, skipped_no_original=skipped_no_original),
                    "per_drug": per_drug,
                },
            )
            continue

        judgments, sb, sn = evaluate_drug_pair(
            orig,
            simp,
            client,
            model_used=model_used,
            system=system,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            delay=args.delay,
        )
        skipped_both_empty += sb
        skipped_no_original += sn

        per_drug.append({"drug_name": drug, "judgments": judgments})
        write_report_atomic(
            args.output,
            {
                "schema_version": "1",
                "model_used": model_used,
                "summary": recompute_summary(
                    per_drug,
                    skipped_both_empty=skipped_both_empty,
                    skipped_no_original=skipped_no_original,
                ),
                "per_drug": per_drug,
            },
        )

    print(f"Wrote evaluation report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
