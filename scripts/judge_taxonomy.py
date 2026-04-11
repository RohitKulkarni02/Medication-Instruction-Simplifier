"""
Shared error taxonomy for the LLM judge (evaluate_labels.py).

Atharv / team: import these constants so prompts and JSON outputs stay aligned.
"""

from __future__ import annotations

from typing import Literal

# Verdict labels (one per original vs simplified field comparison)
PRESERVED = "PRESERVED"
SOFTENED = "SOFTENED"
DROPPED = "DROPPED"

JUDGE_CATEGORIES: tuple[str, ...] = (PRESERVED, SOFTENED, DROPPED)
JudgeVerdict = Literal["PRESERVED", "SOFTENED", "DROPPED"]

JUDGE_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    PRESERVED: "Safety-critical meaning is retained (may be rephrased).",
    SOFTENED: "Safety content is present but weakened, vague, or less prominent than the original.",
    DROPPED: "Safety content is missing, contradicted, or replaced with filler relative to the original.",
}
