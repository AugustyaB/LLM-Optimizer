"""Thorough unit tests for Phase 3: Operational Mode Router & Dataset Synthesis Engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from pcompile.config import MODE_CONFIGS
from pcompile.modes import (
    CompiledPrompt,
    build_architect_prompt,
    build_refine_prompt,
    build_step_prompt,
    compile_prompt,
    infer_coding_edge_cases,
)
from tests.conftest import create_rich_session_log
from training.synthesize_dataset import generate_dataset, validate_dataset


@pytest.fixture
def mock_phase3_environment(tmp_path: Path) -> tuple[Path, Path]:
    """Create a mock project environment with task.md and session telemetry."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # task.md: Phase 1 done, Phase 2 done, Phase 3 active
    (project_dir / "task.md").write_text(
        "# Project Task Tracker\n\n"
        "## Phase 1: Core IO\n- [x] Done (`pcompile/session_selector.py`)\n\n"
        "## Phase 2: Telemetry Engine\n- [x] Done (`pcompile/harvester.py`, `pcompile/evictor.py`)\n\n"
        "## Phase 3: Modes & Dataset\n- [ ] Active (`pcompile/modes.py`, `training/synthesize_dataset.py`)\n",
        encoding="utf-8",
    )

    (project_dir / "implementation_plan.md").write_text(
        "### Phase 1: Core IO\n#### Allowed Working Set — Phase 1\n`pcompile/session_selector.py`\n\n"
        "### Phase 2: Telemetry Engine\n#### Allowed Working Set — Phase 2\n`pcompile/harvester.py`, `pcompile/evictor.py`\n\n"
        "### Phase 3: Modes & Dataset\n#### Allowed Working Set — Phase 3\n`pcompile/modes.py`, `training/synthesize_dataset.py`\n",
        encoding="utf-8",
    )

    session_dir = tmp_path / "session"
    log_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"
    records: list[dict[str, Any]] = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": "Always keep zero-idle VRAM footprint.\nDo not scaffold Phase 4 yet.",
            "tool_calls": [],
        },
        {
            "step_index": 1,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "content": "Phase 2 complete, all 36 tests passing.",
            "tool_calls": [{"name": "read_file", "args": {"Path": "pcompile/harvester.py"}}],
        },
        {
            "step_index": 2,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": "Proceed to Phase 3.",
            "tool_calls": [],
        },
    ]
    create_rich_session_log(log_file, records)

    return project_dir, session_dir


class TestOperationalModes:
    """Test STEP, REFINE, and ARCHITECT mode payload construction and token budgets."""

    def test_step_mode_directive(self, mock_phase3_environment: tuple[Path, Path]) -> None:
        """Verify STEP mode produces compact ~120-word directive with eviction locks and invariants."""
        project_dir, session_dir = mock_phase3_environment

        compiled = build_step_prompt(
            session_dir=session_dir,
            project_dir=project_dir,
            raw_prompt="Proceed to Phase 3",
        )

        assert isinstance(compiled, CompiledPrompt)
        assert compiled.mode == "STEP"
        assert compiled.num_ctx == 2048
        assert compiled.num_predict == 250
        assert compiled.temperature == 0.2

        text = compiled.prompt_text
        assert "### PHASE TRANSITION DIRECTIVE -> PHASE 3" in text
        assert "[EVICT_FROM_WORKING_SET]" in text
        assert "[ACTIVE_PHASE_FILES_ONLY]" in text
        assert "pcompile/modes.py" in text
        assert "[PINNED_USER_INVARIANTS]" in text
        assert any("zero-idle vram" in text.lower() for _ in [1])

        # Verify word count is compact (~100-160 words)
        word_count = len(text.split())
        assert word_count < 220

    def test_refine_mode_specification(self, mock_phase3_environment: tuple[Path, Path]) -> None:
        """Verify REFINE mode produces structured ~300-word prompt specification."""
        project_dir, session_dir = mock_phase3_environment

        compiled = build_refine_prompt(
            session_dir=session_dir,
            project_dir=project_dir,
            raw_prompt="Add streaming JSONL parser with chunked memory buffers",
        )

        assert isinstance(compiled, CompiledPrompt)
        assert compiled.mode == "REFINE"
        assert compiled.num_ctx == 4096
        assert compiled.num_predict == 500
        assert compiled.temperature == 0.2

        text = compiled.prompt_text
        assert "### [ROLE & OBJECTIVE]" in text
        assert "### [INTENT & ARCHITECTURE ALIGNMENT]" in text
        assert "### [INFERRED CODING EDGE CASES]" in text
        assert "Memory & Resource Streaming" in text
        assert "### [CONTEXT SCOPE & FILE EVICTION LOCK]" in text
        assert "[EVICT_FROM_WORKING_SET]" in text
        assert "[ACTIVE_PHASE_FILES_ONLY]" in text
        assert "[PINNED_USER_INVARIANTS]" in text

    def test_architect_mode_blueprint(self, mock_phase3_environment: tuple[Path, Path]) -> None:
        """Verify ARCHITECT mode produces master blueprint prompt enforcing phased invariants."""
        project_dir, session_dir = mock_phase3_environment

        compiled = build_architect_prompt(
            session_dir=session_dir,
            project_dir=project_dir,
            raw_prompt="Design full system architecture",
        )

        assert isinstance(compiled, CompiledPrompt)
        assert compiled.mode == "ARCHITECT"
        assert compiled.num_ctx == 8192
        assert compiled.num_predict == 900
        assert compiled.temperature == 0.2

        text = compiled.prompt_text
        assert "### [ROLE & ARCHITECTURAL MANDATE]" in text
        assert "Allowed_Working_Set" in text
        assert "Evict_On_Complete" in text
        assert "Inferred_Edge_Cases" in text
        assert "Verification_Gate" in text
        assert "[EVICT_FROM_WORKING_SET]" in text
        assert "[ACTIVE_PHASE_FILES_ONLY]" in text

    def test_compile_prompt_dispatcher(self, mock_phase3_environment: tuple[Path, Path]) -> None:
        """Verify compile_prompt dispatches correctly by mode string."""
        project_dir, session_dir = mock_phase3_environment

        c_step = compile_prompt("STEP", session_dir=session_dir, project_dir=project_dir)
        assert c_step.mode == "STEP"

        c_refine = compile_prompt("REFINE", session_dir=session_dir, project_dir=project_dir)
        assert c_refine.mode == "REFINE"

        c_arch = compile_prompt("ARCHITECT", session_dir=session_dir, project_dir=project_dir)
        assert c_arch.mode == "ARCHITECT"

    def test_infer_coding_edge_cases(self) -> None:
        """Verify edge case inference across streaming, VRAM, concurrency, and determinism."""
        ec_stream = infer_coding_edge_cases("Parse large jsonl log file with streaming buffer", ["harvester.py"])
        assert any("Memory & Resource Streaming" in e for e in ec_stream)

        ec_vram = infer_coding_edge_cases("Check GPU memory and keep_alive=0", ["vram_guard.py"])
        assert any("Zero-Idle Memory Footprint" in e for e in ec_vram)

        ec_thread = infer_coding_edge_cases("Worker thread pool with async queue", ["worker.py"])
        assert any("Concurrency & Thread Safety" in e for e in ec_thread)


class TestDatasetSynthesisEngine:
    """Test synthesis and validation of 500 QLoRA training pairs."""

    def test_generate_and_validate_500_pairs(self, tmp_path: Path) -> None:
        """Verify 500 diverse training pairs are synthesized and validated in under 2 seconds."""
        output_file = tmp_path / "dataset.jsonl"

        start_t = time.perf_counter()
        records = generate_dataset(output_path=output_file)
        elapsed = time.perf_counter() - start_t

        assert elapsed < 2.0, f"Synthesis took {elapsed:.3f} s (budget: 2.0 s)"
        assert len(records) == 500
        assert output_file.is_file()

        # Run strict validation
        metrics = validate_dataset(records)
        assert metrics["valid"] is True
        assert metrics["total"] == 500
        assert metrics["step"] == 150
        assert metrics["refine"] == 200
        assert metrics["architect"] == 150
        assert metrics["unique_inputs"] == 500
        assert metrics["unique_outputs"] == 500

    def test_dataset_uniqueness_and_dynamic_edge_cases(self) -> None:
        """Verify that all 500 records have 100% unique inputs and outputs with dynamic edge cases."""
        records = generate_dataset()
        inputs = [r["input"] for r in records]
        outputs = [r["output"] for r in records]

        # 100% unique inputs and outputs
        assert len(set(inputs)) == 500, f"Expected 500 unique inputs, got {len(set(inputs))}"
        assert len(set(outputs)) == 500, f"Expected 500 unique outputs, got {len(set(outputs))}"

        # Verify dynamic edge cases in REFINE mode
        refine_records = [r for r in records if "REFINE mode" in r["instruction"]]
        assert len(refine_records) == 200
        # Check that diverse edge cases are present across REFINE records
        all_edge_case_bullets = [
            line.strip()
            for r in refine_records
            for line in r["output"].splitlines()
            if line.strip().startswith("- ") and "Inferred" not in line
        ]
        # Ensure at least 5 different types of edge cases are generated
        assert len(set(all_edge_case_bullets)) >= 50

    def test_dataset_jsonl_content_and_schema(self, tmp_path: Path) -> None:
        """Verify schema fields and key content markers in generated JSONL file."""
        output_file = tmp_path / "dataset_sample.jsonl"
        records = generate_dataset(output_path=output_file, step_count=2, refine_count=2, architect_count=2)

        assert len(records) == 6
        for r in records:
            assert "instruction" in r
            assert "input" in r
            assert "output" in r
            assert "[PINNED_USER_INVARIANTS]" in r["output"]
