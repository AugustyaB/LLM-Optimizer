"""Hybrid Sandwich: Deterministic post-processor anchoring verified context blocks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pcompile.modes import CompiledPrompt


def apply_hybrid_sandwich(llm_response: str, compiled: CompiledPrompt) -> str:
    """Combine LLM generative output with deterministic CPU-verified working set and invariants.

    Why: LLMs may hallucinate non-existent files or drop critical pinned invariants under
    long contexts. The hybrid sandwich preserves the model's creative architectural/edge-case
    reasoning while strictly replacing or appending the deterministic ground-truth blocks.
    """
    cleaned_response = llm_response.strip()

    # Why: Strip any partial or hallucinated context blocks so ground truth is never corrupted.
    for block_header in (
        r"\[EVICT_FROM_WORKING_SET\][\s\S]*?(?=\n\[|$)",
        r"\[ACTIVE_PHASE_FILES_ONLY\][\s\S]*?(?=\n\[|$)",
        r"\[ACTIVE_FILE_SIGNATURES\][\s\S]*?(?=\n\[|$)",
        r"\[PINNED_USER_INVARIANTS\][\s\S]*?(?=\n\[|$)",
    ):
        cleaned_response = re.sub(block_header, "", cleaned_response).strip()

    sandwich_blocks: list[str] = [
        cleaned_response,
        "",
        "### [CONTEXT SCOPE & FILE EVICTION LOCK]",
        compiled.eviction_result.render_block(),
    ]

    # Append active file signatures if present
    if getattr(compiled, "file_signatures", ""):
        sandwich_blocks.append("")
        sandwich_blocks.append(compiled.file_signatures.strip())

    # Append pinned user invariants strictly at the bottom
    sandwich_blocks.append("")
    sandwich_blocks.append("[PINNED_USER_INVARIANTS]")
    if compiled.harvest_result.pinned_invariants:
        for inv in compiled.harvest_result.pinned_invariants:
            sandwich_blocks.append(f"- {inv}")
    else:
        sandwich_blocks.append("- (none)")

    return "\n".join(sandwich_blocks).strip()
