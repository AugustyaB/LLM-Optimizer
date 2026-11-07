"""Operational Mode Router and Prompt Payload Compilers for STEP, REFINE, ARCHITECT, and HANDOFF."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcompile.config import MODE_CONFIGS, ModeConfig
from pcompile.evictor import EvictionResult, evict_working_set_context
from pcompile.harvester import HarvestResult, harvest_session_telemetry
from pcompile.signature_extractor import extract_active_file_signatures


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
    model_input_payload: str = ""
    file_signatures: str = ""

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

    if any(k in combined for k in ("stream", "file", "jsonl", "read", "log", "large", "chunk", "buffer")):
        edge_cases.append(
            "Memory & Resource Streaming: Stream files line-by-line using buffered binary/text reads; "
            "never load multi-gigabyte or uncapped telemetry into a single in-memory list."
        )
    if any(k in combined for k in ("thread", "async", "lock", "process", "concurrent", "worker", "job")):
        edge_cases.append(
            "Concurrency & Thread Safety: Enforce thread-safe state mutations, avoid global mutable side-effects, "
            "and guard atomic writes with temporary file replacement."
        )
    if any(k in combined for k in ("vram", "gpu", "idle", "keep_alive", "purge", "unload", "memory", "cuda")):
        edge_cases.append(
            "Zero-Idle Memory Footprint: Guarantee strict model unloading (keep_alive=0) and purge caches "
            "immediately upon request completion."
        )

    edge_cases.append(
        "Null & Error Boundaries: Gracefully handle missing files, empty directory trees, malformed JSONL records, "
        "and unexpected EOFs with explicit, fail-safe fallbacks."
    )
    edge_cases.append(
        "Non-destructive Code Modifications: Preserve existing unrelated routines, type annotations, and docstrings; "
        "prevent regression of previously passing test suites."
    )
    return edge_cases


def _format_model_input_payload(instruction: str, mode: str, intent: str, harvest: HarvestResult, sigs: str) -> str:
    """Format exact [INSTRUCTION] / [INPUT] payload matching QLoRA fine-tuning format."""
    parts = [f"[MODE: {mode}]", f"User Intent: {intent}", "", harvest.render_prompt_context()]
    if sigs:
        parts.extend(["", sigs])
    return f"[INSTRUCTION]\n{instruction}\n\n[INPUT]\n" + "\n".join(parts)


def build_step_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "Proceed to next phase",
) -> CompiledPrompt:
    """STEP Mode: Rapid ~120-word phase-transition directive."""
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["STEP"]
    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="STEP")
    sigs = extract_active_file_signatures(eviction_res.active_files, root)

    active_num = eviction_res.active_phase_number or 1
    completed_str = ", ".join(f"Phase {p}" for p in eviction_res.completed_phases) or "None"

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
    if sigs:
        lines.extend(["", sigs])
    if harvest_res.pinned_invariants:
        lines.extend(["", "[PINNED_USER_INVARIANTS]", *[f"- {i}" for i in harvest_res.pinned_invariants]])

    payload = _format_model_input_payload("Compile Phase-Transition Directive in STEP mode.", "STEP", raw_prompt, harvest_res, sigs)
    return CompiledPrompt("STEP", "\n".join(lines), config.num_ctx, config.num_predict, 0.2, eviction_res, harvest_res, payload, sigs)


def build_refine_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "",
) -> CompiledPrompt:
    """REFINE Mode: Structured ~300-word prompt specification with dynamic edge cases."""
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["REFINE"]
    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="REFINE")
    sigs = extract_active_file_signatures(eviction_res.active_files, root)

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
        *[f"- {ec}" for ec in edge_cases],
        "",
        "### [CONTEXT SCOPE & FILE EVICTION LOCK]",
        eviction_res.render_block(),
    ]
    if sigs:
        lines.extend(["", sigs])
    if harvest_res.zone_b_turns:
        lines.extend(["", "### [RECENT TELEMETRY & WORKING STATE (ZONE B)]", "\n\n".join(harvest_res.zone_b_turns[-2:])])
    if harvest_res.pinned_invariants:
        lines.extend(["", "[PINNED_USER_INVARIANTS]", *[f"- {i}" for i in harvest_res.pinned_invariants]])

    payload = _format_model_input_payload("Compile prompt specification in REFINE mode.", "REFINE", user_intent, harvest_res, sigs)
    return CompiledPrompt("REFINE", "\n".join(lines), config.num_ctx, config.num_predict, 0.2, eviction_res, harvest_res, payload, sigs)


def build_architect_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "",
) -> CompiledPrompt:
    """ARCHITECT Mode: Comprehensive ~700-word master blueprint compiler prompt."""
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["ARCHITECT"]
    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="ARCHITECT")
    sigs = extract_active_file_signatures(eviction_res.active_files, root)

    intent = raw_prompt.strip() or "Architect a complete, modular, phased implementation plan for the project."
    lines: list[str] = [
        "### [ROLE & ARCHITECTURAL MANDATE]",
        "You are the Principal Autonomous Systems Architect designing a robust, drift-proof system blueprint.",
        f"Mandate: {intent}",
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
    ]
    if sigs:
        lines.extend(["", sigs])
    lines.extend(["", "### [HISTORICAL EXECUTION TELEMETRY]", harvest_res.render_prompt_context()])

    payload = _format_model_input_payload("Compile blueprint in ARCHITECT mode.", "ARCHITECT", intent, harvest_res, sigs)
    return CompiledPrompt("ARCHITECT", "\n".join(lines), config.num_ctx, config.num_predict, 0.2, eviction_res, harvest_res, payload, sigs)


def build_handoff_prompt(
    session_dir: Path,
    project_dir: Path | None = None,
    raw_prompt: str = "",
) -> CompiledPrompt:
    """HANDOFF Mode: Clean-slate session bootstrapper for starting fresh Turn-0 chats with zero lost context."""
    root = project_dir or session_dir
    config: ModeConfig = MODE_CONFIGS["HANDOFF"]
    eviction_res = evict_working_set_context(session_dir=session_dir, project_dir=root)
    harvest_res = harvest_session_telemetry(session_dir=session_dir, mode="HANDOFF")
    sigs = extract_active_file_signatures(eviction_res.active_files, root)

    user_note = raw_prompt.strip() or "Handoff active session state into a clean-slate Turn-0 prompt."
    active_num = eviction_res.active_phase_number or 1
    completed_str = ", ".join(f"Phase {p}" for p in eviction_res.completed_phases) or "None"

    lines: list[str] = [
        "### [HANDOFF & FRESH SESSION BOOTSTRAP DIRECTIVE]",
        f"Completed Phases: {completed_str} (Verified & Closed).",
        f"Target Continuation Phase: Phase {active_num}",
        f"Handoff Directive: {user_note}",
        "",
        "### CONTEXT SCOPE & FILE EVICTION LOCK",
        eviction_res.render_block(),
    ]
    if sigs:
        lines.extend(["", sigs])
    if harvest_res.zone_b_turns:
        lines.extend(["", "### RECENT WORKING STATE & ERRORS (ZONE B)", "\n\n".join(harvest_res.zone_b_turns[-3:])])
    lines.extend(["", "[PINNED_USER_INVARIANTS]"])
    if harvest_res.pinned_invariants:
        lines.extend([f"- {i}" for i in harvest_res.pinned_invariants])
    else:
        lines.append("- (none)")

    payload = _format_model_input_payload("Compile session bootstrapper in HANDOFF mode.", "HANDOFF", user_note, harvest_res, sigs)
    return CompiledPrompt("HANDOFF", "\n".join(lines), config.num_ctx, config.num_predict, 0.2, eviction_res, harvest_res, payload, sigs)


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
    if mode_upper in ("HANDOFF", "-HO", "--HANDOFF"):
        return build_handoff_prompt(session_dir=session_dir, project_dir=project_dir, raw_prompt=raw_prompt)
    return build_refine_prompt(session_dir=session_dir, project_dir=project_dir, raw_prompt=raw_prompt)
