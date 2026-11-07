"""Dataset Synthesis Engine: Hybrid Hugging Face + Dynamic Edge-Case Combinatorial Generator.

Why: Generates 600 high-entropy training records for QLoRA fine-tuning across STEP,
REFINE, ARCHITECT, and HANDOFF modes with zero synthetic numbering in outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pcompile.modes import infer_coding_edge_cases
from training.domain_bank import (
    ARCHITECT_PROMPTS_BANK,
    DOMAINS,
    HANDOFF_PROMPTS_BANK,
    REFINE_PROMPTS_BANK,
)

_STEP_ACTIONS = [
    "Advance execution from Phase {p_from} to Phase {p_to} targeting `{target}` with verification.",
    "Promote implementation from Phase {p_from} to Phase {p_to} locking `{target}` without regressions.",
    "Transition architecture from Phase {p_from} to Phase {p_to} ensuring clean working set in `{target}`.",
    "Migrate components from Phase {p_from} to Phase {p_to} applying strict test gates on `{target}`.",
    "Finalize Phase {p_from} and bootstrap Phase {p_to} with active verification of `{target}`.",
    "Roll out Phase {p_to} deliverables while evicting Phase {p_from} artifacts from `{target}`.",
    "Complete Phase {p_from} sign-off and enter Phase {p_to} focusing on `{target}`.",
]


def load_hf_coding_prompts(count: int = 500) -> list[str]:
    """Attempt to load real diverse coding instructions from Hugging Face if datasets is available."""
    try:
        from datasets import load_dataset  # type: ignore

        for repo_id in ("mlabonne/Evol-Instruct-Python-1k", "nickrosh/Evol-Instruct-Code-80k-v1"):
            try:
                ds = load_dataset(repo_id, split="train", streaming=True)
                prompts: list[str] = []
                for item in ds:
                    val = item.get("instruction") or item.get("prompt") or item.get("text") or ""
                    if isinstance(val, str) and len(val.strip()) > 30:
                        prompts.append(val.strip().replace("\r\n", " "))
                        if len(prompts) >= count:
                            return prompts
            except Exception:
                continue
    except Exception:
        pass
    return []


def synthesize_step_pair(index: int, domain: dict[str, Any]) -> dict[str, Any]:
    """Generate a unique training pair for STEP mode with clean paths and no synthetic numbers."""
    p_from, p_to = (index % 5) + 1, (index % 5) + 2
    topic, evicted, active = domain["topic"], list(domain["files_p1"]), list(domain["files_p2"])
    target = active[index % len(active)]
    action = _STEP_ACTIONS[index % len(_STEP_ACTIONS)].format(p_from=p_from, p_to=p_to, target=target)
    directive = f"{action} ({topic})"

    input_text = (
        f"[MODE: STEP]\nDomain: {topic} (Transition {p_from} -> {p_to}, Variant {index})\n"
        f"Context: Phase {p_from} completed. Target: Phase {p_to}.\n"
        f"User Directive: {directive}\nHarvested Working Set: {', '.join(evicted + active)}"
    )
    output_text = (
        f"### PHASE TRANSITION DIRECTIVE -> PHASE {p_to}\n"
        f"Completed Phases: Phase {p_from} (Verified & Closed).\nActive Directive: {directive}\n\n"
        f"### CONTEXT SCOPE & FILE EVICTION LOCK\n"
        f"[EVICT_FROM_WORKING_SET]\n" + "\n".join(f"- {f}" for f in sorted(evicted)) + "\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n" + "\n".join(f"- {f}" for f in sorted(active)) + "\n\n"
        f"### EXECUTION INVARIANTS\n"
        f"- Scope Lock: Touch ONLY files declared in [ACTIVE_PHASE_FILES_ONLY] for Phase {p_to}.\n"
        f"- Eviction Lock: Do NOT read, modify, or scaffold files listed in [EVICT_FROM_WORKING_SET].\n"
        f"- Verification Gate: Complete all phase tests with 100% pass rate before requesting verification.\n\n"
        f"[PINNED_USER_INVARIANTS]\n" + "\n".join(f"- {inv}" for inv in domain["invariants"])
    )
    return {"instruction": "Compile a drift-proof Phase-Transition Directive for Gemini 3.8 Flash in STEP mode.",
            "input": input_text, "output": output_text}


def synthesize_refine_pair(index: int, domain: dict[str, Any], raw_prompt: str | None = None) -> dict[str, Any]:
    """Generate a unique training pair for REFINE mode with dynamic edge cases and clean paths."""
    prompt_str = raw_prompt or REFINE_PROMPTS_BANK[index % len(REFINE_PROMPTS_BANK)]
    phase_num, topic = (index % 5) + 1, domain["topic"]
    evicted, active = list(domain["files_p1"]), list(domain["files_p2"])
    edge_cases = infer_coding_edge_cases(prompt_str, active) + [
        f"Domain Boundary Safety: Enforce verified runtime constraints for {topic}."
    ]

    input_text = (
        f"[MODE: REFINE]\nDomain: {topic} (Task Variant {index})\nTarget Phase: Phase {phase_num}\n"
        f"User Prompt: {prompt_str}\nActive Working Set: {', '.join(active)}\n"
        f"Stale Files to Evict: {', '.join(evicted)}\nActive Constraints: {'; '.join(domain['invariants'])}"
    )
    output_text = (
        f"### [ROLE & OBJECTIVE]\n"
        f"You are the Lead Autonomous Systems Engineer executing precise codebase modifications.\n"
        f"Primary Objective: {prompt_str}\n\n"
        f"### [INTENT & ARCHITECTURE ALIGNMENT]\n"
        f"- Target Phase: Phase {phase_num} ({topic})\n"
        f"- Architecture Alignment: Enforce zero-idle memory consumption, modularity, and strict context hygiene.\n"
        f"- Scope Guard: Implement changes non-destructively without modifying locked or evicted components.\n\n"
        f"### [INFERRED CODING EDGE CASES]\n" + "\n".join(f"- {ec}" for ec in edge_cases) + "\n\n"
        f"### [CONTEXT SCOPE & FILE EVICTION LOCK]\n"
        f"[EVICT_FROM_WORKING_SET]\n" + "\n".join(f"- {f}" for f in sorted(evicted)) + "\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n" + "\n".join(f"- {f}" for f in sorted(active)) + "\n\n"
        f"[PINNED_USER_INVARIANTS]\n" + "\n".join(f"- {inv}" for inv in domain["invariants"])
    )
    return {"instruction": "Compile a structured feature and bug-fix prompt specification in REFINE mode.",
            "input": input_text, "output": output_text}


def synthesize_architect_pair(index: int, domain: dict[str, Any], raw_mandate: str | None = None) -> dict[str, Any]:
    """Generate a unique training pair for ARCHITECT mode with clean blueprint output."""
    mandate_str = raw_mandate or ARCHITECT_PROMPTS_BANK[index % len(ARCHITECT_PROMPTS_BANK)]
    topic, active, evicted = domain["topic"], list(domain["files_p2"]), list(domain["files_p1"])

    input_text = (
        f"[MODE: ARCHITECT]\nArchitecture Domain: {topic} (Spec Variant {index})\n"
        f"System Mandate: {mandate_str}\nAvailable Subsystems: {', '.join(evicted + active)}\n"
        f"Governing Constraints: {'; '.join(domain['invariants'])}"
    )
    output_text = (
        f"### [ROLE & ARCHITECTURAL MANDATE]\n"
        f"You are the Principal Autonomous Systems Architect designing the system architecture for {topic}.\n"
        f"Mandate: {mandate_str}\n\n"
        f"### [COMPILER DIRECTIVES & PHASE BLUEPRINT SPECIFICATION]\n"
        f"Every phase generated in the architectural plan MUST strictly declare the following four invariants:\n"
        f"1. Allowed_Working_Set: Explicit, comma-separated list of files permitted to be created or edited.\n"
        f"2. Evict_On_Complete: Files from prior phases that must be immediately evicted from context upon phase exit.\n"
        f"3. Inferred_Edge_Cases: Explicit technical constraints (memory streaming, thread safety, boundary checks, zero-idle invariants).\n"
        f"4. Verification_Gate: Exact, executable test commands (e.g. pytest suite) and mandatory criteria required before closing the phase.\n\n"
        f"### [PROJECT CONTEXT & CURRENT PHASE TOPOLOGY]\n"
        f"[EVICT_FROM_WORKING_SET]\n" + "\n".join(f"- {f}" for f in sorted(evicted)) + "\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n" + "\n".join(f"- {f}" for f in sorted(active)) + "\n\n"
        f"[PINNED_USER_INVARIANTS]\n" + "\n".join(f"- {inv}" for inv in domain["invariants"])
    )
    return {"instruction": "Compile a master blueprint prompt for system architecture in ARCHITECT mode.",
            "input": input_text, "output": output_text}


def synthesize_handoff_pair(index: int, domain: dict[str, Any], raw_prompt: str | None = None) -> dict[str, Any]:
    """Generate a unique training pair for HANDOFF mode to reset drift and bootstrap a fresh session."""
    prompt_str = raw_prompt or HANDOFF_PROMPTS_BANK[index % len(HANDOFF_PROMPTS_BANK)]
    topic, turn_count = domain["topic"], 36 + (index % 30)
    evicted, active = list(domain["files_p1"]), list(domain["files_p2"])

    input_text = (
        f"[MODE: HANDOFF]\nDomain: {topic} (Handoff Variant {index})\n"
        f"Turn Count: {turn_count}\nUser Directive: {prompt_str}\n"
        f"Completed Phases: Phase 1 (Verified)\nActive Phase: Phase 2\n"
        f"Active Working Set: {', '.join(active)}\nEvicted Files: {', '.join(evicted)}\n"
        f"Governing Constraints: {'; '.join(domain['invariants'])}"
    )
    output_text = (
        f"### [HANDOFF & FRESH SESSION BOOTSTRAP DIRECTIVE]\n"
        f"You are taking over an in-progress session for {topic} to reset context drift and bootstrap execution.\n"
        f"Handoff Objective: {prompt_str}\n\n"
        f"### [SESSION CONTEXT & PROGRESS SUMMARY]\n"
        f"- Completed Milestones: Phase 1 is fully verified and closed.\n"
        f"- Current State: Execution reached turn {turn_count}; prior context is pruned to prevent drift.\n"
        f"- Fresh Session Invariant: Resume directly from active phase files; do NOT re-execute closed work.\n\n"
        f"### [CONTEXT SCOPE & FILE EVICTION LOCK]\n"
        f"[EVICT_FROM_WORKING_SET]\n" + "\n".join(f"- {f}" for f in sorted(evicted)) + "\n\n"
        f"[ACTIVE_PHASE_FILES_ONLY]\n" + "\n".join(f"- {f}" for f in sorted(active)) + "\n\n"
        f"[PINNED_USER_INVARIANTS]\n" + "\n".join(f"- {inv}" for inv in domain["invariants"])
    )
    return {"instruction": "Compile a clean session handoff and context bootstrap directive in HANDOFF mode.",
            "input": input_text, "output": output_text}


def generate_dataset(
    output_path: Path | str | None = None,
    step_count: int = 150,
    refine_count: int = 200,
    architect_count: int = 150,
    handoff_count: int | None = None,
    use_hf: bool = False,
) -> list[dict[str, Any]]:
    """Generate validated training pairs with 100% unique inputs and outputs."""
    actual_handoff = 100 if handoff_count is None and step_count == 150 and refine_count == 200 and architect_count == 150 else (handoff_count or 0)
    records: list[dict[str, Any]] = []
    hf_prompts = load_hf_coding_prompts(count=refine_count) if use_hf else []

    for idx in range(step_count):
        records.append(synthesize_step_pair(idx, DOMAINS[idx % len(DOMAINS)]))
    for idx in range(refine_count):
        raw_p = hf_prompts[idx] if idx < len(hf_prompts) else None
        records.append(synthesize_refine_pair(idx, DOMAINS[idx % len(DOMAINS)], raw_prompt=raw_p))
    for idx in range(architect_count):
        records.append(synthesize_architect_pair(idx, DOMAINS[idx % len(DOMAINS)]))
    for idx in range(actual_handoff):
        records.append(synthesize_handoff_pair(idx, DOMAINS[idx % len(DOMAINS)]))

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate schema integrity, uniqueness, and mode breakdown."""
    counts = {"step": 0, "refine": 0, "architect": 0, "handoff": 0}
    required = {"instruction", "input", "output"}
    unique_inputs, unique_outputs = set(), set()
    num_pattern = re.compile(r"\b(Task|Constraint|Rule|TRANSITION|blueprint)\s*#\d+", re.IGNORECASE)

    for idx, rec in enumerate(records):
        assert isinstance(rec, dict) and not (required - set(rec.keys())), f"Record {idx} invalid"
        inp, out, inst = rec["input"], rec["output"], rec["instruction"]
        assert inp and out and inst and inp not in unique_inputs and out not in unique_outputs
        assert not num_pattern.search(out), f"Record {idx} contains synthetic numbering in output: {out[:80]}"
        unique_inputs.add(inp)
        unique_outputs.add(out)

        if "STEP mode" in inst:
            counts["step"] += 1
            assert "### PHASE TRANSITION DIRECTIVE" in out and "[EVICT_FROM_WORKING_SET]" in out
        elif "REFINE mode" in inst:
            counts["refine"] += 1
            assert "[ROLE & OBJECTIVE]" in out and "[INFERRED CODING EDGE CASES]" in out
        elif "ARCHITECT mode" in inst:
            counts["architect"] += 1
            assert "Allowed_Working_Set" in out and "Evict_On_Complete" in out
        elif "HANDOFF mode" in inst:
            counts["handoff"] += 1
            assert "[HANDOFF & FRESH SESSION BOOTSTRAP DIRECTIVE]" in out
        else:
            raise ValueError(f"Unknown instruction mode at {idx}: {inst}")

    assert len(unique_inputs) == len(records) and len(unique_outputs) == len(records)
    return {"valid": True, "total": len(records), **counts, "unique_inputs": len(unique_inputs), "unique_outputs": len(unique_outputs)}


def main() -> None:
    """CLI entry point for dataset synthesis and validation."""
    parser = argparse.ArgumentParser(description="Synthesize and validate QLoRA prompt compiler dataset.")
    parser.add_argument("-o", "--output", default="training/dataset.jsonl", help="Target output JSONL path")
    parser.add_argument("--use-hf", action="store_true", help="Fetch prompts from Hugging Face if available")
    args = parser.parse_args()

    start_t = time.perf_counter()
    records = generate_dataset(output_path=args.output, use_hf=args.use_hf)
    metrics = validate_dataset(records)
    elapsed = time.perf_counter() - start_t

    print(f"Dataset generated in {elapsed:.3f}s: {metrics['total']} records (STEP:{metrics['step']}, "
          f"REFINE:{metrics['refine']}, ARCHITECT:{metrics['architect']}, HANDOFF:{metrics['handoff']})")


if __name__ == "__main__":
    main()
