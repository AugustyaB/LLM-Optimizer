"""Operational Mode Router and Prompt Payload Compilers for STEP, REFINE, and ARCHITECT."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcompile.config import MODE_CONFIGS, ModeConfig
from pcompile.evictor import EvictionResult, evict_working_set_context
from pcompile.harvester import HarvestResult, harvest_session_telemetry


@dataclass
class CompiledPrompt:
    """Compiled, drift-proof prompt package ready for LLM consumption."""

    mode: str
    prompt_text: str
    num_ctx: int
    num_predict: int
    temperature: float
    eviction_result: EvictionResult
    harvest_result: HarvestResult

    @property
    def estimated_token_count(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        return max(1, len(self.prompt_text) // 4)


def infer_coding_edge_cases(raw_prompt: str, accessed_files: list[str]) -> list[str]:
    """Deterministically infer unspoken software engineering edge cases based on task intent."""
    edge_cases: list[str] = []
    text_lower = raw_prompt.lower()
    files_str = " ".join(accessed_files).lower()
    combined = f"{text_lower} {files_str}"

    # Streaming / Memory I/O
    if any(k in combined for k in ("stream", "file", "jsonl", "read", "log", "large", "chunk", "buffer")):
        edge_cases.append(
            "Memory & Resource Streaming: Stream files line-by-line using buffered binary/text reads; "
            "never load multi-gigabyte or uncapped telemetry into a single in-memory list."
        )

    # Concurrency / Process safety
    if any(k in combined for k in ("thread", "async", "lock", "process", "concurrent", "worker", "job")):
        edge_cases.append(
            "Concurrency & Thread Safety: Enforce thread-safe state mutations, avoid global mutable side-effects, "
            "and guard atomic writes with temporary file replacement."
        )

    # Zero-idle / VRAM / Hardware
    if any(k in combined for k in ("vram", "gpu", "idle", "keep_alive", "purge", "unload", "memory", "cuda")):
        edge_cases.append(
            "Zero-Idle Memory Footprint: Guarantee strict model unloading (keep_alive=0) and purge caches "
            "immediately upon request completion."
        )

    # Boundary conditions / Nulls
    edge_cases.append(
        "Null & Error Boundaries: Gracefully handle missing files, empty directory trees, malformed JSONL records, "
        "and unexpected EOFs with explicit, fail-safe fallbacks."
    )

    # Non-destructive diffing
    edge_cases.append(
        "Non-destructive Code Modifications: Preserve existing unrelated routines, type annotations, and docstrings; "
        "prevent regression of previously passing test suites."
    )

    # Determinism
    if any(k in combined for k in ("sort", "order", "list", "dataset", "hash", "dict")):
        edge_cases.append(
            "Deterministic Outputs: Ensure all set collections, dictionaries, and file traversals are sorted "
            "to produce 100% reproducible, drift-proof artifacts."
        )

    return edge_cases


def build_step_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "Proceed to next phase",
) -> CompiledPrompt:
    """STEP Mode: Rapid ~120-word phase-transition directive.

    num_ctx: 2048, num_predict: 250.
    Bypasses redundant code verification; inspects task/plan state, applies working-set locks,
    and anchors pinned invariants.
    """
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["STEP"]

    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="STEP")

    active_num = eviction_res.active_phase_number or 1
    completed_str = ", ".join(f"Phase {p}" for p in eviction_res.completed_phases) or "None"

    # Assemble concise Phase-Transition Directive
    lines: list[str] = [
        f"### PHASE TRANSITION DIRECTIVE -> PHASE {active_num}",
        f"Completed Phases: {completed_str} (Verified & Closed).",
        f"Active Directive: {raw_prompt.strip()}",
        "",
        "### CONTEXT SCOPE & FILE EVICTION LOCK",
        eviction_res.render_block(),
        "",
        "### EXECUTION INVARIANTS",
        f"- Scope Lock: Touch ONLY files declared in [ACTIVE_PHASE_FILES_ONLY] for Phase {active_num}.",
        "- Eviction Lock: Do NOT read, modify, or scaffold files listed in [EVICT_FROM_WORKING_SET].",
        "- Verification Gate: Complete all phase tests with 100% pass rate before requesting verification.",
    ]

    if harvest_res.pinned_invariants:
        lines.append("")
        lines.append("[PINNED_USER_INVARIANTS]")
        for inv in harvest_res.pinned_invariants:
            lines.append(f"- {inv}")

    prompt_text = "\n".join(lines)

    return CompiledPrompt(
        mode="STEP",
        prompt_text=prompt_text,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        temperature=0.2,
        eviction_result=eviction_res,
        harvest_result=harvest_res,
    )


def build_refine_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "",
) -> CompiledPrompt:
    """REFINE Mode: Structured ~300-word prompt specification.

    num_ctx: 4096, num_predict: 500.
    Synthesizes intent alignment, inferred coding edge cases, file eviction locks,
    and bottom-anchored user invariants.
    """
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["REFINE"]

    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="REFINE")

    user_intent = raw_prompt.strip() or "Execute active phase feature enhancements and refine implementation."
    edge_cases = infer_coding_edge_cases(user_intent, eviction_res.active_files)
    active_phase = eviction_res.active_phase_number or 1

    lines: list[str] = [
        "### [ROLE & OBJECTIVE]",
        "You are the Lead Autonomous Systems Engineer executing precise codebase modifications.",
        f"Primary Objective: {user_intent}",
        "",
        "### [INTENT & ARCHITECTURE ALIGNMENT]",
        f"- Target Phase: Phase {active_phase}",
        "- Architecture Alignment: Enforce zero-idle memory consumption, modularity, and strict context hygiene.",
        "- Scope Guard: Implement changes non-destructively without modifying locked or evicted components.",
        "",
        "### [INFERRED CODING EDGE CASES]",
    ]
    for ec in edge_cases:
        lines.append(f"- {ec}")

    lines.append("")
    lines.append("### [CONTEXT SCOPE & FILE EVICTION LOCK]")
    lines.append(eviction_res.render_block())

    # Include recent Zone B context if available
    if harvest_res.zone_b_turns:
        lines.append("")
        lines.append("### [RECENT TELEMETRY & WORKING STATE (ZONE B)]")
        lines.append("\n\n".join(harvest_res.zone_b_turns[-2:]))

    # Pinned user invariants anchored at the bottom
    if harvest_res.pinned_invariants:
        lines.append("")
        lines.append("[PINNED_USER_INVARIANTS]")
        for inv in harvest_res.pinned_invariants:
            lines.append(f"- {inv}")

    prompt_text = "\n".join(lines)

    return CompiledPrompt(
        mode="REFINE",
        prompt_text=prompt_text,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        temperature=0.2,
        eviction_result=eviction_res,
        harvest_result=harvest_res,
    )


def build_architect_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "",
) -> CompiledPrompt:
    """ARCHITECT Mode: Comprehensive ~700-word master blueprint compiler prompt.

    num_ctx: 8192, num_predict: 900.
    Forces target AI to construct or audit a phased implementation plan where every phase declares
    Allowed_Working_Set, Evict_On_Complete, Inferred_Edge_Cases, and Verification_Gate.
    """
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["ARCHITECT"]

    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="ARCHITECT")

    system_intent = raw_prompt.strip() or "Architect a complete, modular, phased implementation plan for the project."

    lines: list[str] = [
        "### [ROLE & ARCHITECTURAL MANDATE]",
        "You are the Principal Autonomous Systems Architect designing a robust, drift-proof system blueprint.",
        f"Mandate: {system_intent}",
        "",
        "### [COMPILER DIRECTIVES & PHASE BLUEPRINT SPECIFICATION]",
        "Every phase generated in the architectural plan MUST strictly declare the following four invariants:",
        "1. Allowed_Working_Set: Explicit, comma-separated list of files permitted to be created or edited.",
        "2. Evict_On_Complete: Files from prior phases that must be immediately evicted from context upon phase exit.",
        "3. Inferred_Edge_Cases: Explicit technical constraints (memory streaming, thread safety, boundary checks, zero-idle invariants).",
        "4. Verification_Gate: Exact, executable test commands (e.g. pytest suite) and mandatory criteria required before closing the phase.",
        "",
        "### [PROJECT CONTEXT & CURRENT PHASE TOPOLOGY]",
        eviction_res.render_block(),
        "",
        "### [HISTORICAL EXECUTION TELEMETRY]",
        harvest_res.render_prompt_context(),
    ]

    prompt_text = "\n".join(lines)

    return CompiledPrompt(
        mode="ARCHITECT",
        prompt_text=prompt_text,
        num_ctx=config.num_ctx,
        num_predict=config.num_predict,
        temperature=0.2,
        eviction_result=eviction_res,
        harvest_result=harvest_res,
    )


def compile_prompt(
    mode: str,
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "",
) -> CompiledPrompt:
    """Dispatch prompt compilation according to chosen operational mode."""
    mode_upper = mode.upper().strip()
    if mode_upper in ("STEP", "-ST", "--STEP"):
        return build_step_prompt(session_dir=session_dir, project_dir=project_dir, raw_prompt=raw_prompt)
    if mode_upper in ("ARCHITECT", "ARCH", "-A", "--ARCH"):
        return build_architect_prompt(session_dir=session_dir, project_dir=project_dir, raw_prompt=raw_prompt)
    return build_refine_prompt(session_dir=session_dir, project_dir=project_dir, raw_prompt=raw_prompt)
