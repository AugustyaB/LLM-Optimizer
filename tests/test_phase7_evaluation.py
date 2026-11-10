"""Comprehensive unit tests for Phase 7: Zero-Waste Trimming, Anti-Overcorrection, and 4-Gate Evaluator.

Verifies:
1. Deep HF Solution Constraint Extractor (extracts concrete edge cases from solutions).
2. load_or_generate_dataset force-regeneration on use_hf=True.
3. apply_hybrid_sandwich reasoning tag (<think>...</think>) stripping and deterministic block attachment.
4. Zero-waste dataset synthesis (no CPU blocks in LLM outputs; 20% clean-state Minimal Intervention Lock).
5. Schema validation & fallback circuit breaker (validate_llm_output rejecting filler/short/invalid).
6. Hot-file tracker and signature prioritization ([RECENTLY_MODIFIED_HOT_FILES]).
7. Step mode instant CPU fast-path (bypasses Ollama unless --force-llm).
8. Zone A consecutive ledger deduplication (collapse consecutive matching turns to ... (xN)).
9. evaluate_model 4 verification gates and scorecard generation.
10. pcompile --compare side-by-side execution and clipboard piping.
11. Strict anti-monolith verification (<280 lines across all pcompile/ and training/ modules).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pcompile.cli import build_arg_parser, main, query_ephemeral_ollama
from pcompile.evictor import EvictionResult, evict_working_set_context
from pcompile.harvester import (
    TurnRecord,
    compute_telemetry_health,
    deduplicate_ledger_lines,
    harvest_session_telemetry,
)
from pcompile.modes import compile_prompt
from pcompile.sandwich import apply_hybrid_sandwich, validate_llm_output
from pcompile.signature_extractor import extract_active_file_signatures
from tests.conftest import create_rich_session_log
from training.domain_bank import DOMAINS
from training.evaluate_model import (
    UNSEEN_TEST_PROMPTS,
    print_scorecard,
    run_full_evaluation,
    verify_gate1_loss_and_artifacts,
    verify_gate2_structural_audit,
    verify_gate3_neural_value_add,
    verify_gate4_vram_purge,
)
from training.hf_extractor import extract_solution_constraints, load_hf_pairs_with_constraints
from training.synthesize_dataset import (
    generate_dataset,
    synthesize_architect_pair,
    synthesize_handoff_pair,
    synthesize_refine_pair,
    synthesize_step_pair,
    validate_dataset,
)
from training.train_qlora import load_or_generate_dataset


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
# 2. Dataset Force-Regeneration on --use-hf & Zero-Waste Outputs
# ---------------------------------------------------------------------------
class TestDatasetRegenerationAndZeroWaste:
    """Verify force-regeneration on use_hf=True and zero-waste output formatting."""

    def test_use_hf_forces_regeneration(self, tmp_path: Path) -> None:
        """Existing dataset on disk must be overwritten when use_hf=True."""
        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{"instruction":"old","input":"old","output":"old"}\n', encoding="utf-8")
        assert dataset_file.is_file()

        records = load_or_generate_dataset(dataset_file, use_hf=True)
        assert len(records) == 600
        first_line = json.loads(dataset_file.read_text(encoding="utf-8").splitlines()[0])
        assert first_line["input"] != "old"

    def test_zero_waste_outputs_have_no_deterministic_blocks(self) -> None:
        """Target outputs in dataset must NOT contain [EVICT_FROM_WORKING_SET] or [PINNED_USER_INVARIANTS]."""
        records = generate_dataset()
        metrics = validate_dataset(records)
        assert metrics["valid"] is True

        for rec in records:
            out = rec["output"]
            inst = rec["instruction"]
            if "REFINE mode" in inst or "ARCHITECT mode" in inst or "HANDOFF mode" in inst:
                assert "[EVICT_FROM_WORKING_SET]" not in out
                assert "\n[ACTIVE_PHASE_FILES_ONLY]\n" not in out
                assert "[PINNED_USER_INVARIANTS]" not in out


# ---------------------------------------------------------------------------
# 3. Telemetry Health, Anti-Overcorrection & 20% Clean-State Pairs
# ---------------------------------------------------------------------------
class TestTelemetryHealthAndAntiOvercorrection:
    """Test telemetry health signal, Minimal Intervention Lock, and Anti-Overcorrection Guard."""

    def test_compute_telemetry_health_clean_vs_error(self) -> None:
        """compute_telemetry_health correctly distinguishes clean logs from error traces."""
        clean_turns = [
            TurnRecord(0, "USER", "USER_INPUT", "DONE", "Build parser"),
            TurnRecord(1, "MODEL", "tool", "DONE", "edit_file completed"),
        ]
        assert compute_telemetry_health(clean_turns) == "CLEAN_PASSING"

        error_turns = [
            TurnRecord(0, "MODEL", "tool", "ERROR", "Command failed"),
        ]
        assert compute_telemetry_health(error_turns) == "ERROR_DETECTED"

        trace_turns = [
            TurnRecord(0, "MODEL", "tool", "DONE", "Traceback (most recent call last):\nValueError: bad"),
        ]
        assert compute_telemetry_health(trace_turns) == "ERROR_DETECTED"

    def test_20_percent_clean_state_minimal_intervention_pairs(self) -> None:
        """Every 5th refine pair (index % 5 == 0) must emit Minimal Intervention Lock."""
        d = DOMAINS[0]
        # index 0: clean
        clean_pair = synthesize_refine_pair(0, d)
        assert "[TELEMETRY_HEALTH: CLEAN_PASSING]" in clean_pair["input"]
        assert "Minimal Intervention Lock" in clean_pair["output"]
        assert "Anti-Overcorrection Guard" in clean_pair["output"]

        # index 1: non-clean
        dirty_pair = synthesize_refine_pair(1, d)
        assert "Minimal Intervention Lock" not in dirty_pair["output"]

    def test_clean_health_injects_anti_overcorrection_in_refine_prompt(self, tmp_path: Path) -> None:
        """build_refine_prompt injects Anti-Overcorrection Guard when telemetry is clean."""
        sess = tmp_path / "sess"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Refine test", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Plan\n- [ ] Phase 1\n", encoding="utf-8")
        compiled = compile_prompt("REFINE", sess, project_dir=tmp_path, raw_prompt="Refine feature")

        assert compiled.harvest_result.telemetry_health == "CLEAN_PASSING"
        assert "Anti-Overcorrection Guard" in compiled.prompt_text
        assert "Zero-Scan Lock" in compiled.prompt_text


# ---------------------------------------------------------------------------
# 4. Schema Validation & Circuit Breaker Tests
# ---------------------------------------------------------------------------
class TestSchemaValidationAndCircuitBreaker:
    """Verify validate_llm_output and fallback mechanism in query_ephemeral_ollama."""

    def test_validate_llm_output_valid_response(self) -> None:
        """Structured responses with ### [ header and no filler must pass."""
        valid_resp = (
            "### [ROLE & OBJECTIVE]\n"
            "You are the Lead Autonomous Systems Engineer.\n"
            "### [INTENT & ARCHITECTURE ALIGNMENT]\n"
            "- Modularity and clean boundaries."
        )
        assert validate_llm_output(valid_resp, mode="REFINE") is True

    def test_validate_llm_output_rejects_empty_and_short(self) -> None:
        """Empty or under 40 char responses must fail validation."""
        assert validate_llm_output("", mode="REFINE") is False
        assert validate_llm_output("Too short response", mode="REFINE") is False

    def test_validate_llm_output_rejects_conversational_filler(self) -> None:
        """Responses starting with 'Sure', 'Here is', 'Certainly', 'I can' must fail."""
        for filler in ("Sure! Here is the prompt: ### [ROLE & OBJECTIVE]\n1234567890",
                       "Certainly, below is your plan: ### [ROLE & OBJECTIVE]\n1234567890",
                       "Here is the result you wanted: ### [ROLE & OBJECTIVE]\n1234567890",
                       "I can help with that: ### [ROLE & OBJECTIVE]\n1234567890"):
            assert validate_llm_output(filler, mode="REFINE") is False

    def test_validate_llm_output_rejects_missing_markdown_header(self) -> None:
        """Responses without required markdown headers must fail."""
        resp = "This response is long enough to pass length checks, but lacks any markdown headers entirely."
        assert validate_llm_output(resp, mode="REFINE") is False

    def test_circuit_breaker_falls_back_to_cpu_prompt(self, tmp_path: Path) -> None:
        """query_ephemeral_ollama must fall back to compiled.prompt_text when LLM fails validation."""
        sess = tmp_path / "sess"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Rule 1", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Plan\n- [ ] Phase 1\n", encoding="utf-8")
        compiled = compile_prompt("REFINE", sess, project_dir=tmp_path, raw_prompt="Fix error")

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "Sure, here is your response!"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                out = query_ephemeral_ollama(compiled)

        assert "[FALLBACK: LLM output failed schema check; using verified CPU prompt]" in mock_err.getvalue()
        assert out == compiled.prompt_text


# ---------------------------------------------------------------------------
# 5. Hot-File Tracker & Signature Prioritization
# ---------------------------------------------------------------------------
class TestHotFileTrackerAndSignaturePrioritization:
    """Verify recently modified files tracking and signature extractor ordering."""

    def test_recently_modified_files_in_eviction_result(self, tmp_path: Path) -> None:
        """EvictionResult tracks files modified in last 5 turns and renders [RECENTLY_MODIFIED_HOT_FILES]."""
        res = EvictionResult(
            active_files=["core.py", "utils.py"],
            evicted_files=["old.py"],
            active_phase_number=2,
            completed_phases=[1],
            recently_modified_files=["core.py"],
        )
        rendered = res.render_block()
        assert "[RECENTLY_MODIFIED_HOT_FILES]" in rendered
        assert "- core.py" in rendered

    def test_extract_active_file_signatures_prioritizes_hot_files(self, tmp_path: Path) -> None:
        """extract_active_file_signatures places recently_modified_files at the top."""
        (tmp_path / "f1.py").write_text("def fn1(): pass\n", encoding="utf-8")
        (tmp_path / "f2.py").write_text("def fn2(): pass\n", encoding="utf-8")

        # Prioritize f2 over f1
        sigs = extract_active_file_signatures(
            active_files=["f1.py", "f2.py"],
            root_dir=tmp_path,
            recently_modified_files=["f2.py"],
        )
        # f2 must appear before f1
        idx_f2 = sigs.find("f2.py")
        idx_f1 = sigs.find("f1.py")
        assert idx_f2 < idx_f1


# ---------------------------------------------------------------------------
# 6. Step Mode CPU Fast-Path
# ---------------------------------------------------------------------------
class TestCPUStepFastPath:
    """Verify pcompile --step executes instantly on CPU fast-path without Ollama."""

    def test_step_mode_bypasses_ollama_by_default(self, tmp_path: Path) -> None:
        """pcompile --step must not invoke query_ephemeral_ollama unless --force-llm is set."""
        sess = tmp_path / "brain" / "sess-fast"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Rule 1", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Task\n- [ ] Phase 1\n", encoding="utf-8")

        with patch("pcompile.cli.query_ephemeral_ollama") as mock_ollama:
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO):
                    ret = main([
                        "--step",
                        "--brain-dir", str(tmp_path / "brain"),
                        "--no-clipboard",
                    ])
        assert ret == 0
        mock_ollama.assert_not_called()

    def test_step_mode_invokes_ollama_with_force_llm(self, tmp_path: Path) -> None:
        """pcompile --step --force-llm must invoke query_ephemeral_ollama."""
        sess = tmp_path / "brain" / "sess-force"
        create_rich_session_log(sess / ".system_generated" / "logs" / "transcript.jsonl", [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE", "content": "Rule 1", "tool_calls": []}
        ])
        (sess / "task.md").write_text("# Task\n- [ ] Phase 1\n", encoding="utf-8")

        with patch("pcompile.cli.query_ephemeral_ollama", return_value="### PHASE TRANSITION DIRECTIVE -> PHASE 2") as mock_ollama:
            with patch("sys.stdout", new_callable=io.StringIO):
                with patch("sys.stderr", new_callable=io.StringIO):
                    ret = main([
                        "--step",
                        "--force-llm",
                        "--brain-dir", str(tmp_path / "brain"),
                        "--no-clipboard",
                    ])
        assert ret == 0
        mock_ollama.assert_called_once()


# ---------------------------------------------------------------------------
# 7. Zone A Ledger Consecutive Deduplication
# ---------------------------------------------------------------------------
class TestZoneALedgerDeduplication:
    """Verify collapsing consecutive identical/matching Zone A ledger lines."""

    def test_deduplicate_consecutive_identical_lines(self) -> None:
        """Consecutive identical lines must be collapsed into '<line> (xN)'."""
        lines = [
            "T1 [MODEL]: tool -> edit_file(a.py) (DONE)",
            "T1 [MODEL]: tool -> edit_file(a.py) (DONE)",
            "T1 [MODEL]: tool -> edit_file(a.py) (DONE)",
            "T2 [MODEL]: tool -> run_command(pytest) (DONE)",
        ]
        collapsed = deduplicate_ledger_lines(lines)
        assert len(collapsed) == 2
        assert collapsed[0] == "T1 [MODEL]: tool -> edit_file(a.py) (DONE) (x3)"
        assert collapsed[1] == "T2 [MODEL]: tool -> run_command(pytest) (DONE)"

    def test_deduplicate_consecutive_matching_actions(self) -> None:
        """Consecutive identical actions with incrementing step indices must be collapsed."""
        lines = [
            "T10 [MODEL]: tool -> edit_file(b.py) (DONE)",
            "T11 [MODEL]: tool -> edit_file(b.py) (DONE)",
        ]
        collapsed = deduplicate_ledger_lines(lines)
        assert len(collapsed) == 1
        assert "(x2)" in collapsed[0]


# ---------------------------------------------------------------------------
# 8. Hybrid Sandwich <think> Tag Stripping & Deterministic Attachment
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
        assert "[RECENTLY_MODIFIED_HOT_FILES]" in sandwiched


# ---------------------------------------------------------------------------
# 9. Model Verification & Scorecard Suite
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

    def test_gate2_structural_audit(self) -> None:
        """Gate 2 verifies structure, zero preamble, and clean-pass restraint."""
        d = DOMAINS[1]

        def _clean_mock(prompt: str) -> str:
            if "STEP" in prompt:
                return synthesize_step_pair(0, d)["output"]
            if "ARCHITECT" in prompt:
                return synthesize_architect_pair(0, d)["output"]
            if "HANDOFF" in prompt:
                return synthesize_handoff_pair(0, d)["output"]
            if "CLEAN_PASSING" in prompt:
                return synthesize_refine_pair(0, d)["output"]
            return synthesize_refine_pair(1, d)["output"]

        passed, detail, outputs = verify_gate2_structural_audit("pcompile", "http://localhost", mock_fn=_clean_mock)
        assert passed is True
        assert len(outputs) == 5

    def test_gate3_novel_technical_token_ratio(self) -> None:
        """Gate 3 calculates novel technical tokens beyond CPU baseline."""
        refine_out = (
            "### [ROLE & OBJECTIVE]\nOptimize OrderService.\n\n"
            "### [INFERRED CODING EDGE CASES]\n"
            "- Idempotency Token Store: Cache payment idempotency UUIDs in Redis cluster with TTL lease.\n"
            "- Circuit Breaker: Trip circuit breaker upon 3 consecutive gateway HTTP 504 timeouts.\n\n"
        )
        passed, detail, ratio = verify_gate3_neural_value_add(refine_out, "Implement payment timeout rollback")
        assert passed is True
        assert ratio >= 0.15

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
            if "CLEAN_PASSING" in prompt:
                return synthesize_refine_pair(0, d)["output"]
            res = synthesize_refine_pair(1, d)["output"]
            return res.replace(
                "[INFERRED CODING EDGE CASES]",
                "[INFERRED CODING EDGE CASES]\n- Distributed Mutex: Acquire Redlock token before initiating transaction.\n- Exponential Jitter: Apply full jitter backoff loop.",
            )

        res = run_full_evaluation(output_dir=out, mock_fn=_mock_model, mock_vram_safe=True)
        assert res["overall_passed"] is True


# ---------------------------------------------------------------------------
# 10. CLI --compare Flag Verification
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
# 11. Anti-Monolith Verification
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
