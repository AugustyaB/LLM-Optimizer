"""Comprehensive unit tests for Phase 7: Deep HF Extractor & LLM Verification Suite.

Verifies:
1. Deep HF Solution Constraint Extractor (extracts concrete edge cases from solutions).
2. load_or_generate_dataset force-regeneration on use_hf=True.
3. apply_hybrid_sandwich reasoning tag (<think>...</think>) stripping.
4. evaluate_model 4 verification gates and scorecard generation.
5. pcompile --compare side-by-side execution and clipboard piping.
6. Strict anti-monolith verification (<280 lines across all modules).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pcompile.cli import main
from pcompile.modes import compile_prompt
from pcompile.sandwich import apply_hybrid_sandwich
from tests.conftest import create_rich_session_log
from training.domain_bank import DOMAINS
from training.evaluate_model import (
    print_scorecard,
    run_full_evaluation,
    verify_gate1_loss_and_artifacts,
    verify_gate2_structural_audit,
    verify_gate3_neural_value_add,
    verify_gate4_vram_purge,
)
from training.hf_extractor import extract_solution_constraints, load_hf_pairs_with_constraints
from training.synthesize_dataset import (
    synthesize_architect_pair,
    synthesize_handoff_pair,
    synthesize_refine_pair,
    synthesize_step_pair,
)
from training.train_qlora import QLoRAConfig, load_or_generate_dataset, run_smoke_test


# ---------------------------------------------------------------------------
# 1. HF Extractor Tests
# ---------------------------------------------------------------------------
class TestHFSolutionConstraintExtractor:
    """Test extracting concrete technical edge cases from reference code solutions."""

    def test_extract_exception_and_transaction_constraints(self) -> None:
        """Exceptions and atomic transactions in code must be captured as constraints."""
        code = (
            "async def transfer_funds(account_id: str, amount: float) -> bool:\n"
            "    try:\n"
            "        async with db.transaction():\n"
            "            await account.debit(amount)\n"
            "            db.commit()\n"
            "    except (TimeoutError, ConnectionRefusedError) as err:\n"
            "        db.rollback()\n"
            "        raise AccountSyncError(str(err))\n"
        )
        inst = "Implement secure funds transfer with timeout rollback."
        constraints = extract_solution_constraints(inst, code)
        text = " ".join(constraints)

        assert "Exception Safety" in text
        assert "TimeoutError" in text or "AccountSyncError" in text
        assert "Transactional Isolation" in text
        assert "Concurrency Guard" in text

    def test_extract_memory_and_validation_constraints(self) -> None:
        """Chunked generators and input checks must produce memory bounds and validation."""
        code = (
            "def stream_large_file(path: str, chunk_size: int = 4096):\n"
            "    assert isinstance(path, str) and len(path) > 0\n"
            "    with open(path, 'rb') as f:\n"
            "        while chunk := f.read(chunk_size):\n"
            "            yield chunk\n"
        )
        inst = "Stream large binary records with low memory footprint."
        constraints = extract_solution_constraints(inst, code)
        text = " ".join(constraints)

        assert "Memory Bound" in text
        assert "Input Validation" in text

    def test_extract_constraints_fallback_on_empty(self) -> None:
        """Brief or empty code must yield at least 2 safe fallback constraints."""
        constraints = extract_solution_constraints("Simple utility function", "return x + y")
        assert len(constraints) >= 2
        assert "Algorithmic Efficiency" in constraints[0]

    def test_load_hf_pairs_streaming_mock(self) -> None:
        """Mock dataset streaming and check record structure."""
        mock_items = [
            {
                "instruction": "Optimize database index scan",
                "output": "def scan_index(): try: pass\nexcept KeyError: pass",
            }
        ]
        mock_ds_mod = MagicMock()
        mock_ds_mod.load_dataset.return_value = mock_items
        with patch.dict("sys.modules", {"datasets": mock_ds_mod}):
            records = load_hf_pairs_with_constraints(count=1)
            assert len(records) == 1
            assert "instruction" in records[0]
            assert "output_code" in records[0]
            assert len(records[0]["constraints"]) >= 1


# ---------------------------------------------------------------------------
# 2. Dataset Force-Regeneration on --use-hf
# ---------------------------------------------------------------------------
class TestDatasetRegeneration:
    """Verify that passing use_hf=True regenerates dataset even if file exists on disk."""

    def test_use_hf_forces_regeneration(self, tmp_path: Path) -> None:
        """Existing dataset on disk must be overwritten when use_hf=True."""
        dataset_file = tmp_path / "dataset.jsonl"
        # Write dummy line
        dataset_file.write_text('{"instruction":"old","input":"old","output":"old"}\n', encoding="utf-8")
        assert dataset_file.is_file()

        records = load_or_generate_dataset(dataset_file, use_hf=True)
        assert len(records) == 600
        # Check that file content was overwritten
        first_line = json.loads(dataset_file.read_text(encoding="utf-8").splitlines()[0])
        assert first_line["input"] != "old"


# ---------------------------------------------------------------------------
# 3. Hybrid Sandwich <think> Tag Stripping
# ---------------------------------------------------------------------------
class TestHybridSandwichThinkStripping:
    """Verify apply_hybrid_sandwich strips reasoning tags before anchoring locks."""

    def test_strip_non_empty_think_tags(self, tmp_path: Path) -> None:
        """Leading <think>...</think> reasoning blocks must be cleanly removed."""
        sess = tmp_path / "sess"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Rule 1", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Plan\n- [ ] Phase 1\n", encoding="utf-8")
        compiled = compile_prompt("REFINE", sess, project_dir=tmp_path, raw_prompt="Fix parser")

        llm_output = (
            "<think>\n"
            "The user needs a clean parser fix. Let me consider memory limits.\n"
            "</think>\n"
            "### [ROLE & OBJECTIVE]\n"
            "Fix parser memory leaks."
        )
        sandwiched = apply_hybrid_sandwich(llm_output, compiled)

        assert "<think>" not in sandwiched
        assert "</think>" not in sandwiched
        assert "The user needs a clean parser fix." not in sandwiched
        assert "### [ROLE & OBJECTIVE]\nFix parser memory leaks." in sandwiched
        assert "[ACTIVE_PHASE_FILES_ONLY]" in sandwiched
        assert "[PINNED_USER_INVARIANTS]" in sandwiched

    def test_strip_empty_think_tags(self, tmp_path: Path) -> None:
        """Empty <think>\n\n</think> tags must be cleanly removed."""
        sess = tmp_path / "sess2"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Rule 1", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Plan\n- [ ] Phase 1\n", encoding="utf-8")
        compiled = compile_prompt("STEP", sess, project_dir=tmp_path, raw_prompt="Proceed")

        llm_output = "<think>\n\n</think>### PHASE TRANSITION DIRECTIVE -> PHASE 2"
        sandwiched = apply_hybrid_sandwich(llm_output, compiled)

        assert "<think>" not in sandwiched
        assert "</think>" not in sandwiched
        assert "### PHASE TRANSITION DIRECTIVE -> PHASE 2" in sandwiched


# ---------------------------------------------------------------------------
# 4. Model Verification & A/B Benchmark Suite
# ---------------------------------------------------------------------------
class TestModelEvaluationSuite:
    """Test 4 verification gates and scorecard generation."""

    def test_gate1_loss_and_artifacts(self, tmp_path: Path) -> None:
        """Gate 1 verifies Modelfile, gguf export, and loss convergence."""
        out = tmp_path / "outputs"
        out.mkdir()
        (out / "Modelfile").write_text("FROM test\n", encoding="utf-8")
        (out / "gguf").mkdir()
        (out / "gguf" / "model.gguf").write_text("dummy", encoding="utf-8")
        (out / "training_metrics.json").write_text(
            json.dumps({"initial_loss": 2.4, "final_loss": 0.35, "total_steps": 350}), encoding="utf-8"
        )

        passed, detail = verify_gate1_loss_and_artifacts(out)
        assert passed is True
        assert "Initial: 2.400" in detail

    def test_gate1_fails_on_diverged_loss(self, tmp_path: Path) -> None:
        """Gate 1 must fail if final_loss >= initial_loss."""
        out = tmp_path / "outputs"
        out.mkdir()
        (out / "Modelfile").write_text("FROM test\n", encoding="utf-8")
        (out / "gguf").mkdir()
        (out / "gguf" / "model.gguf").write_text("dummy", encoding="utf-8")
        (out / "training_metrics.json").write_text(
            json.dumps({"initial_loss": 0.5, "final_loss": 1.8, "total_steps": 350}), encoding="utf-8"
        )

        passed, detail = verify_gate1_loss_and_artifacts(out)
        assert passed is False
        assert "Loss did not converge" in detail

    def test_gate2_structural_audit(self) -> None:
        """Gate 2 verifies structure, zero preamble, and no <think> tags across 4 unseen prompts."""
        d = DOMAINS[1]

        def _clean_mock(prompt: str) -> str:
            if "STEP" in prompt:
                return synthesize_step_pair(0, d)["output"]
            if "ARCHITECT" in prompt:
                return synthesize_architect_pair(0, d)["output"]
            if "HANDOFF" in prompt:
                return synthesize_handoff_pair(0, d)["output"]
            return synthesize_refine_pair(0, d)["output"]

        passed, detail, outputs = verify_gate2_structural_audit("pcompile", "http://localhost", mock_fn=_clean_mock)
        assert passed is True
        assert len(outputs) == 4

    def test_gate2_fails_on_conversational_preamble(self) -> None:
        """Gate 2 must fail if model emits chatty preamble."""
        d = DOMAINS[1]

        def _chatty_mock(prompt: str) -> str:
            return "Sure! Here is the prompt specification:\n" + synthesize_refine_pair(0, d)["output"]

        passed, detail, _ = verify_gate2_structural_audit("pcompile", "http://localhost", mock_fn=_chatty_mock)
        assert passed is False
        assert "Found conversational preamble" in detail

    def test_gate3_novel_technical_token_ratio(self) -> None:
        """Gate 3 calculates novel technical tokens beyond CPU baseline."""
        refine_out = (
            "### [ROLE & OBJECTIVE]\nOptimize OrderService.\n\n"
            "### [INFERRED CODING EDGE CASES]\n"
            "- Idempotency Token Store: Cache payment idempotency UUIDs in Redis cluster with TTL lease.\n"
            "- Circuit Breaker: Trip circuit breaker upon 3 consecutive gateway HTTP 504 timeouts.\n\n"
            "[ACTIVE_PHASE_FILES_ONLY]\n- OrderService.java\n"
        )
        passed, detail, ratio = verify_gate3_neural_value_add(refine_out, "Implement payment timeout rollback")
        assert passed is True
        assert ratio >= 0.15
        assert "novel tokens" in detail

    def test_gate4_vram_purge_check(self) -> None:
        """Gate 4 checks that VRAM is purged."""
        passed, _ = verify_gate4_vram_purge(mock_vram_safe=True)
        assert passed is True

    def test_full_evaluation_scorecard_flow(self, tmp_path: Path) -> None:
        """Execute full evaluation suite and verify scorecard."""
        out = tmp_path / "outputs"
        out.mkdir()
        (out / "Modelfile").write_text("FROM test\n", encoding="utf-8")
        (out / "gguf").mkdir()
        (out / "gguf" / "model.gguf").write_text("dummy", encoding="utf-8")
        (out / "training_metrics.json").write_text(
            json.dumps({"initial_loss": 2.2, "final_loss": 0.4, "total_steps": 350}), encoding="utf-8"
        )

        d = DOMAINS[1]

        def _mock_model(prompt: str) -> str:
            if "STEP" in prompt:
                return synthesize_step_pair(0, d)["output"]
            if "ARCHITECT" in prompt:
                return synthesize_architect_pair(0, d)["output"]
            if "HANDOFF" in prompt:
                return synthesize_handoff_pair(0, d)["output"]
            res = synthesize_refine_pair(0, d)["output"]
            return res.replace(
                "[INFERRED CODING EDGE CASES]",
                "[INFERRED CODING EDGE CASES]\n- Distributed Mutex: Acquire Redlock token before initiating transaction.\n- Exponential Jitter: Apply full jitter backoff loop.",
            )

        res = run_full_evaluation(output_dir=out, mock_fn=_mock_model, mock_vram_safe=True)
        assert res["overall_passed"] is True
        assert res["gate1_loss_artifacts"]["passed"] is True
        assert res["gate2_structural_audit"]["passed"] is True
        assert res["gate3_neural_value_add"]["passed"] is True
        assert res["gate4_vram_purge"]["passed"] is True


# ---------------------------------------------------------------------------
# 5. CLI --compare Flag Verification
# ---------------------------------------------------------------------------
class TestCLICompareFlag:
    """Verify pcompile --compare output formatting and clipboard pipe."""

    def test_cli_compare_flag_output(self, tmp_path: Path) -> None:
        """--compare must output both deterministic CPU baseline and neural output."""
        sess = tmp_path / "brain" / "sess-cmp"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Rule 1", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Task\n- [ ] Phase 1\n", encoding="utf-8")

        with patch("pcompile.cli.query_ephemeral_ollama", return_value="### [ROLE & OBJECTIVE]\nNeural Output"):
            with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                with patch("sys.stderr", new_callable=io.StringIO):
                    ret = main([
                        "--compare",
                        "--brain-dir", str(tmp_path / "brain"),
                        "--no-clipboard",
                        "Optimize query indexing",
                    ])

        assert ret == 0
        output = mock_out.getvalue()
        assert "[1] DETERMINISTIC CPU BASELINE (--dry-harvest)" in output
        assert "[2] TRAINED QWEN3-8B NEURAL OUTPUT (Hybrid Sandwich)" in output
        assert "Neural Output" in output


# ---------------------------------------------------------------------------
# 6. Anti-Monolith Verification
# ---------------------------------------------------------------------------
class TestPhase7AntiMonolith:
    """Verify that NO Python module in pcompile/ or training/ exceeds 280 lines."""

    def test_all_modules_under_280_lines(self) -> None:
        """Every single module must remain strictly under 280 lines."""
        root = Path(__file__).resolve().parent.parent
        oversized: list[tuple[str, int]] = []
        for folder in ("pcompile", "training"):
            for py_file in (root / folder).rglob("*.py"):
                lines = len(py_file.read_text(encoding="utf-8").splitlines())
                if lines > 280:
                    oversized.append((py_file.name, lines))

        assert not oversized, f"The following modules exceed 280 lines: {oversized}"
