from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterable

from prompt_templates import SIMPLIFY_SYSTEM_PROMPT, SIMPLIFY_USER_PROMPT_TEMPLATE

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"


def _load_dotenv() -> None:
    """Load repo-root .env into os.environ (no extra dependency)."""
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


@dataclass(frozen=True)
class SimplificationResult:
    drug_name: str
    original_text: str
    simplified_text: str
    boxed_warning: str
    dosage: str
    warnings: list[str]
    contraindications: list[str]
    interactions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug_name": self.drug_name,
            "original_text": self.original_text,
            "simplified_text": self.simplified_text,
            "boxed_warning": self.boxed_warning,
            "dosage": self.dosage,
            "warnings": self.warnings,
            "contraindications": self.contraindications,
            "interactions": self.interactions,
        }


def _read_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    if path.endswith(".jsonl"):
        items: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
        return obj["items"]
    raise ValueError(f"Unsupported JSON structure for input: {path}")


def _write_json(path: str, items: list[dict[str, Any]], pretty: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(items, f, ensure_ascii=False, indent=2)
        else:
            json.dump(items, f, ensure_ascii=False)


def _coerce_list(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    if isinstance(x, str):
        # Common structured encodings: newline-separated or semicolon-separated.
        s = x.strip()
        if not s:
            return []
        s = s.replace("\n", "; ")
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts
    # Fallback (keep as string).
    s = str(x).strip()
    return [s] if s else []


def _strip(s: str | None) -> str:
    return (s or "").strip()


def _extract_between_markers(text: str, start: str, end: str | None) -> str:
    """
    Best-effort extraction for semi-structured labels.
    This is intentionally simple; the “real” pipeline should rely on upstream extraction.
    """
    if not text:
        return ""
    flags = re.IGNORECASE | re.DOTALL
    # e.g. start marker "DOSAGE" and stop marker "WARNINGS"
    if end:
        pattern = rf"{re.escape(start)}\s*:?(.*?)\s*{re.escape(end)}\s*:?"
        m = re.search(pattern, text, flags=flags)
        if m:
            return m.group(1).strip()
        return ""
    pattern = rf"{re.escape(start)}\s*:?(.*)$"
    m = re.search(pattern, text, flags=flags)
    return m.group(1).strip() if m else ""


def extract_sections(item: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize input into a consistent set of fields:
    - boxed_warning (string)
    - dosage (string)
    - warnings (list[str])
    - contraindications (list[str])
    - interactions (list[str])
    - original_text (string)
    """
    original_text = _strip(item.get("original_text")) or _strip(item.get("full_label")) or _strip(item.get("label_text"))

    boxed_warning = _strip(item.get("boxed_warning"))

    dosage = _strip(
        item.get("dosage")
        or item.get("dosage_section")
        or item.get("dosage_instructions")
        or item.get("dosage_and_administration")
    )

    warnings = _coerce_list(item.get("warnings") or item.get("warnings_section"))
    contraindications = _coerce_list(item.get("contraindications") or item.get("contraindications_section"))
    interactions = _coerce_list(
        item.get("interactions") or item.get("interactions_section") or item.get("drug_interactions")
    )

    # Per-field fallbacks from a single label blob (openFDA full_label often lacks separate dosage key in JSON).
    if original_text:
        if not boxed_warning:
            boxed_warning = _strip(
                _extract_between_markers(original_text, "BOXED WARNING", "WARNINGS")
                or _extract_between_markers(original_text, "BOXED WARNING", "DOSAGE AND ADMINISTRATION")
                or _extract_between_markers(original_text, "BOXED WARNING", "DOSAGE")
            )
        if not dosage:
            dosage = _strip(
                _extract_between_markers(original_text, "DOSAGE AND ADMINISTRATION", "WARNINGS")
                or _extract_between_markers(original_text, "DOSAGE AND ADMINISTRATION", "CONTRAINDICATIONS")
                or _extract_between_markers(original_text, "DOSAGE AND ADMINISTRATION", "DRUG INTERACTIONS")
                or _extract_between_markers(original_text, "DOSAGE", "WARNINGS")
                or _extract_between_markers(original_text, "DOSAGE", "CONTRAINDICATIONS")
            )

    # Fallback when almost everything is missing: parse whole blob.
    if not any([dosage, warnings, contraindications, interactions, boxed_warning]) and original_text:
        dosage = _strip(
            dosage
            or _extract_between_markers(original_text, "DOSAGE AND ADMINISTRATION", "WARNINGS")
            or _extract_between_markers(original_text, "DOSAGE", "WARNINGS")
        )
        if not warnings:
            w = _extract_between_markers(original_text, "WARNINGS", "CONTRAINDICATIONS")
            warnings = _coerce_list(w)
        if not contraindications:
            c = _extract_between_markers(original_text, "CONTRAINDICATIONS", "DRUG INTERACTIONS")
            contraindications = _coerce_list(c)
        if not interactions:
            i = _extract_between_markers(original_text, "DRUG INTERACTIONS", None)
            interactions = _coerce_list(i)

    return {
        "original_text": original_text,
        "boxed_warning": boxed_warning,
        "dosage": dosage,
        "warnings": warnings,
        "contraindications": contraindications,
        "interactions": interactions,
    }


def simplify_local(item: dict[str, Any]) -> SimplificationResult:
    sections = extract_sections(item)
    drug_name = _strip(item.get("drug_name") or item.get("brand_name") or "unknown_drug")

    boxed_warning = _strip(sections["boxed_warning"])
    dosage = _strip(sections["dosage"])
    warnings = sections["warnings"]
    contraindications = sections["contraindications"]
    interactions = sections["interactions"]
    original_text = _strip(sections["original_text"])

    # Local “simplification” is mainly reformatting to a consistent structure.
    # It intentionally does not try to paraphrase away safety content.
    lines: list[str] = []
    lines.append(f"Patient-friendly medication instructions for {drug_name}")
    lines.append("")
    if boxed_warning:
        lines.append("Boxed warning")
        lines.append(boxed_warning)
        lines.append("")
    if dosage:
        lines.append("Dosage")
        lines.append(dosage)
        lines.append("")
    if warnings:
        lines.append("Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    if contraindications:
        lines.append("Contraindications")
        for c in contraindications:
            lines.append(f"- {c}")
        lines.append("")
    if interactions:
        lines.append("Drug interactions")
        for i in interactions:
            lines.append(f"- {i}")
        lines.append("")

    # If we failed to populate any structured fields, keep the original text as fallback.
    simplified_text = "\n".join(lines).strip()
    if not simplified_text and original_text:
        simplified_text = original_text.strip()

    return SimplificationResult(
        drug_name=drug_name,
        original_text=original_text,
        simplified_text=simplified_text,
        boxed_warning=boxed_warning,
        dosage=dosage,
        warnings=warnings,
        contraindications=contraindications,
        interactions=interactions,
    )


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Extract the first balanced JSON object from a string.
    Useful when a model returns surrounding text despite instructions.
    """
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


def _structured_sufficient_for_llm(
    boxed_in: str,
    dosage: str,
    warnings: list[str],
    contraindications: list[str],
    interactions: list[str],
) -> bool:
    """If True, skip pasting full original_label text (avoids duplicating huge SPL blobs)."""
    if dosage and (boxed_in or warnings or contraindications or interactions):
        return True
    if boxed_in and dosage:
        return True
    parts = [bool(dosage), bool(boxed_in), bool(warnings), bool(contraindications), bool(interactions)]
    return sum(1 for p in parts if p) >= 3


def _truncate_for_llm(text: str, max_chars: int) -> str:
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 32)].rstrip() + "\n[... truncated for API size ...]"


def _build_drug_label_blob_for_llm(
    *,
    drug_name: str,
    boxed_in: str,
    dosage: str,
    warnings: list[str],
    contraindications: list[str],
    interactions: list[str],
    original_text: str,
    max_chars: int,
) -> str:
    structured = (
        (f"Drug name: {drug_name}\n" if drug_name else "")
        + (f"BOXED WARNING:\n{boxed_in}\n\n" if boxed_in else "")
        + (f"DOSAGE:\n{dosage}\n\n" if dosage else "")
        + ("WARNINGS:\n" + "\n".join(warnings) + "\n\n" if warnings else "")
        + ("CONTRAINDICATIONS:\n" + "\n".join(contraindications) + "\n\n" if contraindications else "")
        + ("DRUG INTERACTIONS:\n" + "\n".join(interactions) + "\n\n" if interactions else "")
    ).strip()

    use_full_original = original_text.strip() and not _structured_sufficient_for_llm(
        boxed_in, dosage, warnings, contraindications, interactions
    )

    if use_full_original:
        # Sparse structured fields: send truncated narrative label only.
        head = f"Drug name: {drug_name}\n\n" if drug_name else ""
        ot = _truncate_for_llm(original_text, max(1000, max_chars - len(head) - 64))
        blob = f"{head}Original label:\n{ot}".strip()
    else:
        # Enough structured sections: do not paste full_label again (saves tokens on Groq/OpenAI).
        blob = structured

    return _truncate_for_llm(blob, max_chars) if max_chars > 0 else blob


def simplify_openai_compatible(
    item: dict[str, Any],
    *,
    model: str,
    api_key: str,
    temperature: float,
    base_url: str | None = None,
    max_label_chars: int | None = None,
) -> SimplificationResult:
    """
    Chat Completions with JSON output. Use base_url for OpenAI-compatible providers (e.g. Groq).
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "openai package is required for API simplification. pip install openai"
        ) from e

    sections = extract_sections(item)
    drug_name = _strip(item.get("drug_name") or item.get("brand_name") or "unknown_drug")

    # Build a “structured blob” for the prompt.
    boxed_in = sections["boxed_warning"]
    dosage = sections["dosage"]
    warnings = sections["warnings"]
    contraindications = sections["contraindications"]
    interactions = sections["interactions"]
    original_text = sections["original_text"]

    cap = max_label_chars
    if cap is None:
        # Groq free tier can reject huge prompts; keep default conservative (override with env or --max-llm-chars).
        cap = int(os.environ.get("SIMPLIFY_MAX_LLM_LABEL_CHARS", "8000"))
    if cap <= 0:
        cap = 8000

    drug_label_blob = _build_drug_label_blob_for_llm(
        drug_name=drug_name,
        boxed_in=boxed_in,
        dosage=dosage,
        warnings=warnings,
        contraindications=contraindications,
        interactions=interactions,
        original_text=original_text,
        max_chars=cap,
    )

    user_prompt = SIMPLIFY_USER_PROMPT_TEMPLATE.format(drug_label=drug_label_blob)

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Note: response_format is supported for many models; if unavailable it will throw,
    # so we keep parsing robust as well.
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SIMPLIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SIMPLIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        data = _extract_first_json_object(raw)

    # Validate/normalize the output shape to our expected schema.
    boxed_out = _strip(data.get("boxed_warning") or "") or boxed_in
    dosage_out = _strip(data.get("dosage") or "")
    warnings_out = _coerce_list(data.get("warnings") or [])
    contraindications_out = _coerce_list(data.get("contraindications") or [])
    interactions_out = _coerce_list(data.get("interactions") or [])
    simplified_text_out = _strip(data.get("simplified_text") or "")

    if not simplified_text_out:
        # Fallback to keep pipeline moving if the model omits combined text.
        parts: list[str] = [
            f"Patient-friendly medication instructions for {drug_name}".strip(),
            "",
        ]
        if boxed_out:
            parts.extend(["Boxed warning", boxed_out, ""])
        parts.extend(
            [
                "Dosage",
                dosage_out,
                "",
                "Warnings",
                "\n".join(f"- {w}" for w in warnings_out),
                "",
                "Contraindications",
                "\n".join(f"- {c}" for c in contraindications_out),
                "",
                "Drug interactions",
                "\n".join(f"- {i}" for i in interactions_out),
            ]
        )
        simplified_text_out = "\n".join(parts).strip()

    return SimplificationResult(
        drug_name=drug_name,
        original_text=original_text,
        simplified_text=simplified_text_out,
        boxed_warning=boxed_out,
        dosage=dosage_out,
        warnings=warnings_out,
        contraindications=contraindications_out,
        interactions=interactions_out,
    )


def simplify_openai(
    item: dict[str, Any],
    *,
    model: str,
    api_key: str,
    temperature: float,
    max_label_chars: int | None = None,
) -> SimplificationResult:
    """OpenAI platform (api.openai.com)."""
    return simplify_openai_compatible(
        item,
        model=model,
        api_key=api_key,
        temperature=temperature,
        base_url=None,
        max_label_chars=max_label_chars,
    )


def simplify_groq(
    item: dict[str, Any],
    *,
    model: str,
    api_key: str,
    temperature: float,
    max_label_chars: int | None = None,
) -> SimplificationResult:
    """Groq via OpenAI-compatible API (`https://api.groq.com/openai/v1`)."""
    base = os.environ.get("GROQ_OPENAI_BASE_URL", GROQ_OPENAI_BASE_URL).strip() or GROQ_OPENAI_BASE_URL
    return simplify_openai_compatible(
        item,
        model=model,
        api_key=api_key,
        temperature=temperature,
        base_url=base,
        max_label_chars=max_label_chars,
    )


def _load_sample_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "raw_labels", "sample_labels.json")
    )


def self_test() -> int:
    sample_path = _load_sample_path()
    items = _read_json_or_jsonl(sample_path)
    if not items:
        print("Self-test failed: sample input is empty.", file=sys.stderr)
        return 1

    # Local mode should preserve safety content verbatim as strings.
    for item in items:
        res = simplify_local(item)

        sections = extract_sections(item)
        dosage = sections["dosage"]
        for w in sections["warnings"]:
            if w and w not in res.simplified_text:
                print(f"Self-test failed: warning not preserved for {res.drug_name}.", file=sys.stderr)
                return 1
        for c in sections["contraindications"]:
            if c and c not in res.simplified_text:
                print(
                    f"Self-test failed: contraindication not preserved for {res.drug_name}.",
                    file=sys.stderr,
                )
                return 1
        for i in sections["interactions"]:
            if i and i not in res.simplified_text:
                print(f"Self-test failed: interaction not preserved for {res.drug_name}.", file=sys.stderr)
                return 1
        if dosage and dosage not in res.simplified_text:
            print(f"Self-test failed: dosage not preserved for {res.drug_name}.", file=sys.stderr)
            return 1
        bw = sections.get("boxed_warning", "")
        if bw and bw not in res.simplified_text:
            print(f"Self-test failed: boxed warning not preserved for {res.drug_name}.", file=sys.stderr)
            return 1

    print("Self-test passed (local simplification preserves provided safety strings).")
    return 0


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Medication label simplification (local, OpenAI, or Groq OpenAI-compatible API)."
    )
    parser.add_argument("--input", type=str, help="Path to JSON/JSONL input items.")
    parser.add_argument("--output", type=str, required=False, help="Path to write JSON output.")
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["local", "openai", "groq"],
        help="Simplification backend (Groq uses official OpenAI Python SDK + GROQ_API_KEY).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model id (OpenAI or Groq per provider). Default: OPENAI_MODEL or gpt-4o-mini; "
        "GROQ_SIMPLIFY_MODEL or llama-3.1-8b-instant for --provider groq.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--max-llm-chars",
        type=int,
        default=0,
        help="Max characters for label text sent to the LLM (0 = use SIMPLIFY_MAX_LLM_LABEL_CHARS env or default).",
    )
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--pretty", action="store_true", help="Pretty-print output JSON.")
    parser.add_argument("--sleep-ms", type=int, default=0, help="Sleep between items (avoid rate limits).")
    parser.add_argument("--self-test", action="store_true", help="Run a quick local test on sample labels.")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.input or not args.output:
        print("Error: --input and --output are required unless --self-test is used.", file=sys.stderr)
        return 2

    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()

    if args.provider == "openai" and not openai_key:
        print("Error: OPENAI_API_KEY is not set. Use --provider local/groq or set the key.", file=sys.stderr)
        return 2
    if args.provider == "groq" and not groq_key:
        print("Error: GROQ_API_KEY is not set. Use --provider local/openai or set the key.", file=sys.stderr)
        return 2

    if args.model.strip():
        model = args.model.strip()
    elif args.provider == "groq":
        model = os.environ.get("GROQ_SIMPLIFY_MODEL", "llama-3.1-8b-instant").strip()
    else:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

    items = _read_json_or_jsonl(args.input)
    if args.max_items > 0:
        items = items[: args.max_items]

    max_llm_chars: int | None = args.max_llm_chars if args.max_llm_chars > 0 else None

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        try:
            if args.provider == "local":
                res = simplify_local(item)
            elif args.provider == "openai":
                res = simplify_openai(
                    item,
                    model=model,
                    api_key=openai_key,
                    temperature=args.temperature,
                    max_label_chars=max_llm_chars,
                )
            else:
                res = simplify_groq(
                    item,
                    model=model,
                    api_key=groq_key,
                    temperature=args.temperature,
                    max_label_chars=max_llm_chars,
                )
            results.append(res.to_dict())
        except Exception as e:
            print(f"[{idx}] Simplification failed for item={item.get('drug_name')}: {e}", file=sys.stderr)
            raise

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    _write_json(args.output, results, pretty=args.pretty)
    print(f"Wrote {len(results)} simplified items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

