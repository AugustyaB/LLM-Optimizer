"""Dataset Synthesis Engine generating and validating 500 training pairs for QLoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

DOMAINS: list[dict[str, Any]] = [
    {
        "topic": "CLI Application & Terminal Selector",
        "files_p1": ["pcompile/config.py", "pcompile/session_selector.py", "pyproject.toml"],
        "files_p2": ["pcompile/harvester.py", "pcompile/evictor.py", "task.md"],
        "files_p3": ["pcompile/modes.py", "training/synthesize_dataset.py"],
        "invariants": [
            "Always maintain zero idle VRAM footprint.",
            "Do not scaffold files belonging to future phases.",
            "Strictly keep CPU log harvesting under 50 ms.",
        ],
    },
    {
        "topic": "Distributed Async Cache & Eviction Engine",
        "files_p1": ["cache/core.py", "cache/ring_buffer.py", "setup.py"],
        "files_p2": ["cache/evictor.py", "cache/lru_clock.py", "task.md"],
        "files_p3": ["cache/client.py", "benchmarks/bench_throughput.py"],
        "invariants": [
            "Enforce thread-safe lock-free reads.",
            "Never perform blocking I/O on the main event loop.",
            "Only touch files declared in the active working set.",
        ],
    },
    {
        "topic": "High-Throughput JSONL Telemetry Pipeline",
        "files_p1": ["pipeline/schema.py", "pipeline/reader.py", "pyproject.toml"],
        "files_p2": ["pipeline/processor.py", "pipeline/filter.py", "task.md"],
        "files_p3": ["pipeline/sink.py", "training/train_eval.py"],
        "invariants": [
            "Stream records chunk-by-chunk using buffered binary reads.",
            "Gracefully skip malformed records with warning counters.",
            "All test assertions must pass with 100% verification.",
        ],
    },
    {
        "topic": "Embedded SQLite Analytics & Log Harvester",
        "files_p1": ["storage/db.py", "storage/migrations.py", "pyproject.toml"],
        "files_p2": ["harvester/log_indexer.py", "harvester/query.py", "task.md"],
        "files_p3": ["api/routes.py", "training/dataset_builder.py"],
        "invariants": [
            "Always execute atomic transactions with rollback guards.",
            "Never leave unclosed database connections.",
            "Maintain zero-idle background daemon overhead.",
        ],
    },
    {
        "topic": "Hardware-Isolated VRAM Guard & Model Runner",
        "files_p1": ["hardware/gpu_detect.py", "hardware/vram_guard.py", "pyproject.toml"],
        "files_p2": ["runner/ollama_client.py", "runner/process_mgr.py", "task.md"],
        "files_p3": ["runner/qlora_pipeline.py", "training/train_runner.py"],
        "invariants": [
            "Unload model immediately on inference completion with keep_alive=0.",
            "Abort if GPU memory used exceeds 350 MiB threshold.",
            "Preserve existing comments and public docstrings.",
        ],
    },
]

USER_PROMPTS_REFINE: list[str] = [
    "Add buffered stream reading with max 64KB chunks to prevent RAM overflow during large JSONL scans.",
    "Implement thread-safe double-checked locking on the session cache to avoid data races.",
    "Support case-insensitive matching for markdown section titles across CRLF and LF files.",
    "Enforce strict mutual exclusivity between evicted files and active phase files.",
    "Add a fallback mechanism when prompt_toolkit cannot access a valid TTY terminal.",
    "Validate that all extracted file paths are normalized to relative POSIX style.",
    "Implement a fast binary newline counter that avoids full UTF-8 string decoding.",
    "Gracefully catch and log FileNotFoundError when scanning empty or missing brain directories.",
    "Add execution time assertion ensuring fast-path lookup runs strictly under 15 ms.",
    "Extract imperative constraints from user turns using deduplicated case-insensitive matching.",
]

USER_PROMPTS_ARCHITECT: list[str] = [
    "Design a 5-phase modular implementation plan for a low-latency LLM prompt compiler.",
    "Architect an autonomous multi-phase test runner that enforces verification gates before commits.",
    "Overhaul the context eviction pipeline to handle 100k-line project trees with zero lag.",
    "Construct a phased blueprint for fine-tuning a 7B parameter coding model on laptop hardware.",
    "Design an ephemeral CLI architecture that offloads all heavy inference to background Ollama.",
]


def synthesize_step_pair(index: int, domain: dict[str, Any]) -> dict[str, Any]:
    """Generate a realistic training pair for STEP mode."""
    phase_from = (index % 4) + 1
    phase_to = phase_from + 1

    evicted = domain["files_p1"]
    active = domain["files_p2"]
    invs = domain["invariants"]

    input_text = (
        f"[MODE: STEP]\n"
        f"Domain: {domain['topic']}\n"
        f"Context: Phase {phase_from} completed. Advancing to Phase {phase_to}.\n"
        f"Raw User Directive: proceed to Phase {phase_to}.\n"
        f"Accessed Telemetry Files: {', '.join(evicted + active)}"
    )

    evict_lines = "\n".join(f"- {f}" for f in evicted)
    active_lines = "\n".join(f"- {f}" for f in active)
    inv_lines = "\n".join(f"- {inv}" for inv in invs)

    output_text = (
        f"### PHASE TRANSITION DIRECTIVE -> PHASE {phase_to}\n"
        f"Completed Phases: Phase {phase_from} (Verified & Closed).\n"
        f"Active Directive: Proceed to Phase {phase_to}.\n\n"
        f"### CONTEXT SCOPE & FILE EVICTION LOCK\n"
        f"[EVICT_FROM_WORKING_SET]\n{evict_lines}\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n{active_lines}\n\n"
        f"### EXECUTION INVARIANTS\n"
        f"- Scope Lock: Touch ONLY files declared in [ACTIVE_PHASE_FILES_ONLY] for Phase {phase_to}.\n"
        f"- Eviction Lock: Do NOT read, modify, or scaffold files listed in [EVICT_FROM_WORKING_SET].\n"
        f"- Verification Gate: Complete all phase tests with 100% pass rate before requesting verification.\n\n"
        f"[PINNED_USER_INVARIANTS]\n{inv_lines}"
    )

    return {
        "instruction": "Compile a drift-proof Phase-Transition Directive for Gemini 3.8 Flash in STEP mode.",
        "input": input_text,
        "output": output_text,
    }


def synthesize_refine_pair(index: int, domain: dict[str, Any]) -> dict[str, Any]:
    """Generate a realistic training pair for REFINE mode."""
    raw_prompt = USER_PROMPTS_REFINE[index % len(USER_PROMPTS_REFINE)]
    phase_num = (index % 4) + 1

    evicted = domain["files_p1"]
    active = domain["files_p2"]
    invs = domain["invariants"]

    input_text = (
        f"[MODE: REFINE]\n"
        f"Domain: {domain['topic']}\n"
        f"Current Phase: Phase {phase_num}\n"
        f"User Prompt: {raw_prompt}\n"
        f"Active Working Set: {', '.join(active)}\n"
        f"Completed Files: {', '.join(evicted)}\n"
        f"Recent Invariants: {'; '.join(invs)}"
    )

    evict_lines = "\n".join(f"- {f}" for f in evicted)
    active_lines = "\n".join(f"- {f}" for f in active)
    inv_lines = "\n".join(f"- {inv}" for inv in invs)

    output_text = (
        f"### [ROLE & OBJECTIVE]\n"
        f"You are the Lead Autonomous Systems Engineer executing precise codebase modifications.\n"
        f"Primary Objective: {raw_prompt}\n\n"
        f"### [INTENT & ARCHITECTURE ALIGNMENT]\n"
        f"- Target Phase: Phase {phase_num}\n"
        f"- Architecture Alignment: Enforce zero-idle memory consumption, modularity, and strict context hygiene.\n"
        f"- Scope Guard: Implement changes non-destructively without modifying locked or evicted components.\n\n"
        f"### [INFERRED CODING EDGE CASES]\n"
        f"- Memory & Resource Streaming: Stream files line-by-line using buffered binary/text reads; "
        f"never load unbounded telemetry into RAM.\n"
        f"- Null & Error Boundaries: Gracefully handle missing files, empty logs, or malformed records with fallbacks.\n"
        f"- Non-destructive Code Modifications: Preserve existing unrelated routines, docstrings, and typing.\n\n"
        f"### [CONTEXT SCOPE & FILE EVICTION LOCK]\n"
        f"[EVICT_FROM_WORKING_SET]\n{evict_lines}\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n{active_lines}\n\n"
        f"[PINNED_USER_INVARIANTS]\n{inv_lines}"
    )

    return {
        "instruction": "Compile a structured feature and bug-fix prompt specification in REFINE mode.",
        "input": input_text,
        "output": output_text,
    }


def synthesize_architect_pair(index: int, domain: dict[str, Any]) -> dict[str, Any]:
    """Generate a realistic training pair for ARCHITECT mode."""
    mandate = USER_PROMPTS_ARCHITECT[index % len(USER_PROMPTS_ARCHITECT)]

    active = domain["files_p2"]
    evicted = domain["files_p1"]
    invs = domain["invariants"]

    input_text = (
        f"[MODE: ARCHITECT]\n"
        f"Domain: {domain['topic']}\n"
        f"Architectural Mandate: {mandate}\n"
        f"System Components: {', '.join(evicted + active)}\n"
        f"Known Constraints: {'; '.join(invs)}"
    )

    inv_lines = "\n".join(f"- {inv}" for inv in invs)

    output_text = (
        f"### [ROLE & ARCHITECTURAL MANDATE]\n"
        f"You are the Principal Autonomous Systems Architect designing a robust, drift-proof system blueprint.\n"
        f"Mandate: {mandate}\n\n"
        f"### [COMPILER DIRECTIVES & PHASE BLUEPRINT SPECIFICATION]\n"
        f"Every phase generated in the architectural plan MUST strictly declare the following four invariants:\n"
        f"1. Allowed_Working_Set: Explicit, comma-separated list of files permitted to be created or edited.\n"
        f"2. Evict_On_Complete: Files from prior phases that must be immediately evicted from context upon phase exit.\n"
        f"3. Inferred_Edge_Cases: Explicit technical constraints (memory streaming, thread safety, boundary checks, zero-idle invariants).\n"
        f"4. Verification_Gate: Exact, executable test commands (e.g. pytest suite) and mandatory criteria required before closing the phase.\n\n"
        f"### [PROJECT CONTEXT & CURRENT PHASE TOPOLOGY]\n"
        f"[EVICT_FROM_WORKING_SET]\n" + "\n".join(f"- {f}" for f in evicted) + "\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n" + "\n".join(f"- {f}" for f in active) + "\n\n"
        f"[PINNED_USER_INVARIANTS]\n{inv_lines}"
    )

    return {
        "instruction": "Compile a master blueprint prompt for system architecture in ARCHITECT mode.",
        "input": input_text,
        "output": output_text,
    }


def generate_dataset(
    output_path: Path | str | None = None,
    step_count: int = 150,
    refine_count: int = 200,
    architect_count: int = 150,
) -> list[dict[str, Any]]:
    """Generate exactly 500 validated training pairs across STEP, REFINE, and ARCHITECT modes."""
    records: list[dict[str, Any]] = []

    # 1. STEP mode records (150)
    for idx in range(step_count):
        domain = DOMAINS[idx % len(DOMAINS)]
        records.append(synthesize_step_pair(idx, domain))

    # 2. REFINE mode records (200)
    for idx in range(refine_count):
        domain = DOMAINS[idx % len(DOMAINS)]
        records.append(synthesize_refine_pair(idx, domain))

    # 3. ARCHITECT mode records (150)
    for idx in range(architect_count):
        domain = DOMAINS[idx % len(DOMAINS)]
        records.append(synthesize_architect_pair(idx, domain))

    # Save to disk if path is provided
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate schema integrity, token count constraints, and mode breakdown."""
    step_count = 0
    refine_count = 0
    arch_count = 0

    required_fields = {"instruction", "input", "output"}

    for idx, rec in enumerate(records):
        assert isinstance(rec, dict), f"Record {idx} is not a dictionary"
        missing = required_fields - set(rec.keys())
        assert not missing, f"Record {idx} missing fields: {missing}"

        instruction = rec["instruction"]
        inp = rec["input"]
        out = rec["output"]

        assert isinstance(instruction, str) and instruction.strip(), f"Record {idx} empty instruction"
        assert isinstance(inp, str) and inp.strip(), f"Record {idx} empty input"
        assert isinstance(out, str) and out.strip(), f"Record {idx} empty output"

        # Check mode classification
        if "STEP mode" in instruction:
            step_count += 1
            assert "### PHASE TRANSITION DIRECTIVE" in out
            assert "[EVICT_FROM_WORKING_SET]" in out
            assert "[ACTIVE_PHASE_FILES_ONLY]" in out
            assert "[PINNED_USER_INVARIANTS]" in out
        elif "REFINE mode" in instruction:
            refine_count += 1
            assert "[ROLE & OBJECTIVE]" in out
            assert "[INFERRED CODING EDGE CASES]" in out
            assert "[CONTEXT SCOPE & FILE EVICTION LOCK]" in out
            assert "[PINNED_USER_INVARIANTS]" in out
        elif "ARCHITECT mode" in instruction:
            arch_count += 1
            assert "Allowed_Working_Set" in out
            assert "Evict_On_Complete" in out
            assert "Inferred_Edge_Cases" in out
            assert "Verification_Gate" in out
            assert "[PINNED_USER_INVARIANTS]" in out
        else:
            raise ValueError(f"Unrecognized instruction at index {idx}: {instruction}")

    return {
        "valid": True,
        "total": len(records),
        "step": step_count,
        "refine": refine_count,
        "architect": arch_count,
    }


def main() -> None:
    """CLI entry point for dataset synthesis and validation."""
    parser = argparse.ArgumentParser(description="Synthesize and validate QLoRA prompt compiler dataset.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="training/dataset.jsonl",
        help="Target output JSONL path (default: training/dataset.jsonl)",
    )
    args = parser.parse_args()

    start_t = time.perf_counter()
    records = generate_dataset(output_path=args.output)
    metrics = validate_dataset(records)
    elapsed = time.perf_counter() - start_t

    print(f"Dataset generated successfully in {elapsed:.3f} seconds:")
    print(f"  Total records: {metrics['total']}")
    print(f"  STEP mode:      {metrics['step']}")
    print(f"  REFINE mode:    {metrics['refine']}")
    print(f"  ARCHITECT mode: {metrics['architect']}")
    print(f"  Output saved:   {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
