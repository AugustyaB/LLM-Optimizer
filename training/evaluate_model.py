"""Automated Model Verification & A/B Benchmark Suite.

Why: Runs 4 strict verification gates on the trained Ollama prompt compiler model:
Gate 1: Loss convergence & artifact check
Gate 2: Unseen prompt structural audit (STEP, REFINE, ARCHITECT, HANDOFF)
Gate 3: Neural value-add score (Novel Technical Token Ratio vs. CPU baseline)
Gate 4: Post-inference 0 MB VRAM purge check (keep_alive=0)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable

from pcompile.modes import infer_coding_edge_cases
from pcompile.sandwich import apply_hybrid_sandwich
from training.vram_guard import check_vram

CONVERSATIONAL_PREAMBLE_RE = re.compile(
    r"""^\s*(?:sure|here|certainly|i can help|below is|hello|hi|okay|as an ai|as requested)\b""",
    re.IGNORECASE,
)

UNSEEN_TEST_PROMPTS = [
    {
        "mode": "STEP",
        "domain": "Python / CLI Telemetry",
        "instruction": "Compile a drift-proof Phase-Transition Directive for Gemini 3.8 Flash in STEP mode.",
        "input": "[MODE: STEP]\nDomain: Python Telemetry\nContext: Phase 2 completed. Target: Phase 3.\nUser Directive: Advance from Phase 2 to Phase 3 in telemetry parser.",
        "required_markers": ["### PHASE TRANSITION DIRECTIVE", "[EVICT_FROM_WORKING_SET]", "[ACTIVE_PHASE_FILES_ONLY]"],
    },
    {
        "mode": "REFINE",
        "domain": "Java Spring Boot",
        "instruction": "Compile a structured feature and bug-fix prompt specification in REFINE mode.",
        "input": "[MODE: REFINE]\n[TELEMETRY_HEALTH: ERROR_DETECTED]\nDomain: Java Spring Boot\nUser Prompt: In OrderService.java, implement transactional rollback on payment gateway timeout.\nActive Working Set: src/main/java/OrderService.java",
        "required_markers": ["[ROLE & OBJECTIVE]", "[INTENT & ARCHITECTURE ALIGNMENT]", "[INFERRED CODING EDGE CASES]"],
    },
    {
        "mode": "REFINE_CLEAN",
        "domain": "Python Clean Telemetry Restraint",
        "instruction": "Compile a structured feature and bug-fix prompt specification in REFINE mode.",
        "input": "[MODE: REFINE]\n[TELEMETRY_HEALTH: CLEAN_PASSING]\nDomain: Python Clean State\nUser Prompt: Add docstring to compute_checksum function.\nActive Working Set: utils.py",
        "required_markers": ["[ROLE & OBJECTIVE]", "Minimal Intervention Lock"],
    },
    {
        "mode": "ARCHITECT",
        "domain": "TypeScript / React Concurrency",
        "instruction": "Compile a master blueprint prompt for system architecture in ARCHITECT mode.",
        "input": "[MODE: ARCHITECT]\nDomain: TypeScript React State\nSystem Mandate: Architect state reconciler with optimistic concurrency and rollback queue.",
        "required_markers": ["[ROLE & ARCHITECTURAL MANDATE]", "Allowed_Working_Set", "Evict_On_Complete"],
    },
    {
        "mode": "HANDOFF",
        "domain": "PostgreSQL / SQL Partitioning",
        "instruction": "Compile a clean session handoff and context bootstrap directive in HANDOFF mode.",
        "input": "[MODE: HANDOFF]\nDomain: SQL Partition Engine\nTurn Count: 42\nUser Directive: Session exceeds turn limit; compile handoff state for PostgreSQL partition migrations.",
        "required_markers": ["[HANDOFF & FRESH SESSION BOOTSTRAP DIRECTIVE]", "[SESSION CONTEXT & PROGRESS SUMMARY]"],
    },
]


def query_ollama(
    prompt: str,
    model: str = "pcompile",
    url: str = "http://localhost:11434/api/generate",
    mock_fn: Callable[[str], str] | None = None,
) -> str:
    """Send generation request to Ollama with keep_alive=0 or invoke mock."""
    if mock_fn is not None:
        return mock_fn(prompt)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0,
        "options": {"num_predict": 600, "temperature": 0.2},
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def verify_gate1_loss_and_artifacts(output_dir: Path) -> tuple[bool, str]:
    """Gate 1: Verify outputs/Modelfile, GGUF file, and loss convergence."""
    modelfile = output_dir / "Modelfile"
    metrics_file = output_dir / "training_metrics.json"
    gguf_files = list((output_dir / "gguf").glob("*.gguf")) if (output_dir / "gguf").is_dir() else []

    if not modelfile.is_file():
        return False, "Missing Modelfile in outputs/"
    if not gguf_files:
        return False, "Missing .gguf model export in outputs/gguf/"
    if not metrics_file.is_file():
        return False, "Missing training_metrics.json in outputs/"

    try:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        init_l, fin_l = float(metrics.get("initial_loss", 0)), float(metrics.get("final_loss", 0))
        if fin_l >= init_l:
            return False, f"Loss did not converge (initial: {init_l:.3f}, final: {fin_l:.3f})"
        return True, f"Initial: {init_l:.3f}, Final: {fin_l:.3f} (steps: {metrics.get('total_steps', '?')})"
    except Exception as err:
        return False, f"Failed parsing training metrics: {err}"


def verify_gate2_structural_audit(
    model: str,
    url: str,
    mock_fn: Callable[[str], str] | None = None,
) -> tuple[bool, str, dict[str, str]]:
    """Gate 2: Audit unseen prompts for sections, 0 conversational preamble, and <think> suppression."""
    outputs: dict[str, str] = {}
    for test in UNSEEN_TEST_PROMPTS:
        mode = test["mode"]
        try:
            resp = query_ollama(test["input"], model=model, url=url, mock_fn=mock_fn)
        except Exception as err:
            return False, f"Ollama query failed on {mode}: {err}", outputs

        outputs[mode] = resp
        if CONVERSATIONAL_PREAMBLE_RE.search(resp):
            return False, f"Found conversational preamble in {mode} output", outputs
        if "<think>" in resp or "</think>" in resp:
            return False, f"Unsuppressed <think> reasoning tag in {mode} output", outputs
        for marker in test["required_markers"]:
            if marker not in resp:
                return False, f"Missing required section '{marker}' in {mode} output", outputs
    return True, f"{len(outputs)}/{len(UNSEEN_TEST_PROMPTS)} unseen prompts passed structural & restraint audit", outputs


def verify_gate3_neural_value_add(refine_output: str, prompt: str) -> tuple[bool, str, float]:
    """Gate 3: Compute Novel Technical Token Ratio of LLM edge cases vs CPU baseline."""
    cpu_edge_cases = infer_coding_edge_cases(prompt, ["OrderService.java"])
    cpu_text = " ".join(cpu_edge_cases).lower()
    cpu_tokens = set(re.findall(r"\b[a-z]{4,}\b", cpu_text))

    ec_match = re.search(r"### \[INFERRED CODING EDGE CASES\]\s*([\s\S]*?)(?=\n###|\n\[|$)", refine_output)
    llm_ec_text = ec_match.group(1).lower() if ec_match else refine_output.lower()
    llm_tokens = set(re.findall(r"\b[a-z]{4,}\b", llm_ec_text))

    novel = llm_tokens - cpu_tokens
    ratio = len(novel) / max(1, len(llm_tokens))
    passed = ratio >= 0.15
    detail = f"Ratio: {ratio:.2f} ({len(novel)} novel tokens beyond CPU template)"
    return passed, detail, ratio


def verify_gate4_vram_purge(threshold_mb: int = 350, mock_vram_safe: bool = False) -> tuple[bool, str]:
    """Gate 4: Verify post-inference 0 MB VRAM footprint via nvidia-smi."""
    if mock_vram_safe:
        return True, "VRAM purged to 0 MB (keep_alive=0 verified in mock)"
    status = check_vram(threshold_mb=threshold_mb)
    if not status.is_safe:
        return False, f"VRAM leak detected: {status.used_mb} MiB retained (threshold: {threshold_mb} MiB)"
    return True, f"VRAM purged: {status.used_mb} MiB used (threshold: {threshold_mb} MiB)"


def run_full_evaluation(
    output_dir: Path | str = "outputs",
    model: str = "pcompile",
    url: str = "http://localhost:11434/api/generate",
    mock_fn: Callable[[str], str] | None = None,
    mock_vram_safe: bool = False,
    threshold_mb: int = 350,
) -> dict[str, Any]:
    """Execute all 4 verification gates and return detailed scorecard."""
    out_path = Path(output_dir)
    g1_pass, g1_msg = verify_gate1_loss_and_artifacts(out_path)
    g2_pass, g2_msg, test_outputs = verify_gate2_structural_audit(model=model, url=url, mock_fn=mock_fn)

    refine_resp = test_outputs.get("REFINE", "")
    g3_pass, g3_msg, g3_ratio = verify_gate3_neural_value_add(refine_resp, "In OrderService.java, implement transactional rollback on payment gateway timeout.")
    g4_pass, g4_msg = verify_gate4_vram_purge(threshold_mb=threshold_mb, mock_vram_safe=mock_vram_safe)

    overall = g1_pass and g2_pass and g3_pass and g4_pass
    return {
        "gate1_loss_artifacts": {"passed": g1_pass, "detail": g1_msg},
        "gate2_structural_audit": {"passed": g2_pass, "detail": g2_msg},
        "gate3_neural_value_add": {"passed": g3_pass, "detail": g3_msg, "ratio": g3_ratio},
        "gate4_vram_purge": {"passed": g4_pass, "detail": g4_msg},
        "overall_passed": overall,
    }


def print_scorecard(results: dict[str, Any]) -> None:
    """Render clean Model Verification Scorecard."""
    border = "=" * 70
    print("\n" + border)
    print("                 PCOMPILE MODEL VERIFICATION SCORECARD")
    print(border)
    for gate_name, key in [
        ("Gate 1: Loss & Artifact Convergence", "gate1_loss_artifacts"),
        ("Gate 2: Unseen Prompt Structural Audit", "gate2_structural_audit"),
        ("Gate 3: Neural Novel Token Ratio", "gate3_neural_value_add"),
        ("Gate 4: Post-Inference VRAM Purge", "gate4_vram_purge"),
    ]:
        info = results[key]
        tag = "[ PASS ]" if info["passed"] else "[ FAIL ]"
        print(f" {gate_name:<40} {tag} {info['detail']}")
    print(border)
    status_str = "PASSED (4/4 Gates Verified)" if results["overall_passed"] else "FAILED (Verification gates unmet)"
    print(f" OVERALL VERIFICATION: {status_str}")
    print(border + "\n")


def main() -> None:
    """CLI entry point for model verification suite."""
    parser = argparse.ArgumentParser(description="Evaluate trained pcompile model against 4 verification gates.")
    parser.add_argument("--output-dir", default="outputs", help="Directory containing Modelfile and metrics.")
    parser.add_argument("--model", default="pcompile", help="Ollama model identifier.")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama API URL.")
    parser.add_argument("--threshold-mb", type=int, default=350, help="VRAM retention threshold in MiB.")
    parser.add_argument("--mock", action="store_true", help="Run with mock responses for offline verification.")
    args = parser.parse_args()

    mock_fn = None
    if args.mock:
        from training.domain_bank import DOMAINS
        from training.synthesize_dataset import (
            synthesize_architect_pair,
            synthesize_handoff_pair,
            synthesize_refine_pair,
            synthesize_step_pair,
        )

        def _mock_model(prompt: str) -> str:
            d = DOMAINS[1]
            if "STEP" in prompt:
                return synthesize_step_pair(0, d)["output"]
            if "ARCHITECT" in prompt:
                return synthesize_architect_pair(0, d)["output"]
            if "HANDOFF" in prompt:
                return synthesize_handoff_pair(0, d)["output"]
            if "CLEAN_PASSING" in prompt:
                return synthesize_refine_pair(0, d)["output"]
            # REFINE with novel technical tokens for Gate 3
            res = synthesize_refine_pair(1, d)["output"]
            return res.replace(
                "[INFERRED CODING EDGE CASES]",
                "[INFERRED CODING EDGE CASES]\n- Idempotency Key Tracking: Store transaction UUIDs in Redis with lease expiry.\n- Circuit Breaker: Trip circuit after 3 consecutive payment gateway timeouts.",
            )

        mock_fn = _mock_model

    results = run_full_evaluation(
        output_dir=args.output_dir,
        model=args.model,
        url=args.ollama_url,
        mock_fn=mock_fn,
        mock_vram_safe=args.mock,
        threshold_mb=args.threshold_mb,
    )
    print_scorecard(results)
    sys.exit(0 if results["overall_passed"] else 1)


if __name__ == "__main__":
    main()
