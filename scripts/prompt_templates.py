"""
Prompt templates for Person 3 (LLM simplification pipeline).

These templates are designed to preserve adherence-critical content:
- dosage instructions
- warnings
- contraindications
- drug interactions
"""

from __future__ import annotations


SIMPLIFY_SYSTEM_PROMPT = (
    "You are a medical instruction simplification assistant. "
    "Rewrite medication instructions into clear, patient-friendly language. "
    "You MUST preserve adherence- and safety-critical information exactly or with equivalent meaning. "
    "Do not omit, soften, or contradict dosage instructions, warnings, contraindications, or drug interactions. "
    "Do not invent new warnings, contraindications, or interactions. "
    "Keep all dose amounts, units, and frequencies unchanged."
)


SIMPLIFY_USER_PROMPT_TEMPLATE = (
    "Simplify the following medication label into patient-friendly language.\n\n"
    "Safety preservation rules (must follow):\n"
    "1) Do not remove dosage instructions.\n"
    "2) Do not remove warnings.\n"
    "3) Do not remove contraindications.\n"
    "4) Do not remove drug interactions.\n"
    "5) Do not soften or weaken warnings/contraindications (keep them as strong safety statements).\n"
    "6) Do not add any new safety facts not present in the input.\n"
    "7) Keep dose amounts/units/frequencies exactly.\n"
    "8) Output must be valid JSON only.\n\n"
    "Label text (structured where available):\n"
    "{drug_label}\n\n"
    "Return JSON with exactly these keys:\n"
    "- dosage: string (patient-friendly dosage instructions)\n"
    "- warnings: array of strings\n"
    "- contraindications: array of strings\n"
    "- interactions: array of strings\n"
    "- simplified_text: string (full combined patient instructions, may include headings)\n"
)

