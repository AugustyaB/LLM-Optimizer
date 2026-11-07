"""Thorough unit tests for Phase 2: CPU Harvester, Sliding Window, Evictor, & Invariant Pinner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pcompile.config import EVICTION_LOOKBACK_TURNS, MAX_TELEMETRY_WINDOW, RECENT_ZONE_WINDOW
from pcompile.evictor import (
    classify_working_set_files,
    evict_working_set_context,
    extract_turn_file_operations,
    parse_plan_phases,
    parse_task_phases,
)
from pcompile.harvester import (
    TurnRecord,
    apply_sliding_window,
    compress_zone_a_turn,
    extract_pinned_invariants,
    format_zone_b_turn,
    harvest_session_telemetry,
    load_transcript_turns,
)
from tests.conftest import create_rich_session_log


class TestSlidingWindowAndTruncation:
    """Tests for 35-turn hard cutoff and two-zone sliding window partitioning."""

    def test_60_turn_log_truncation(self, tmp_path: Path) -> None:
        """A 60-turn log must truncate strictly to 35 turns (Zone A = 30 turns, Zone B = 5 turns)."""
        session_dir = tmp_path / "session_60_turns"
        log_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"

        records: list[dict[str, Any]] = []
        for i in range(60):
            records.append({
                "step_index": i,
                "source": "USER_EXPLICIT" if i % 2 == 0 else "MODEL",
                "type": "USER_INPUT" if i % 2 == 0 else "PLANNER_RESPONSE",
                "status": "DONE",
                "content": f"Turn message number {i}",
                "tool_calls": [] if i % 2 == 0 else [{"name": "read_file", "args": {"Path": f"file_{i}.py"}}],
            })
        create_rich_session_log(log_file, records)

        turns = load_transcript_turns(session_dir)
        assert len(turns) == 60

        zone_a, zone_b = apply_sliding_window(turns, max_window=MAX_TELEMETRY_WINDOW, recent_window=RECENT_ZONE_WINDOW)

        # Invariant verification: strictly 35 turns total
        assert len(zone_a) + len(zone_b) == 35
        # Zone A has exactly 30 turns (N-35 to N-6)
        assert len(zone_a) == 30
        # Zone B has exactly 5 turns (N-5 to N)
        assert len(zone_b) == 5

        # Check turn index bounds
        assert zone_a[0].step_index == 25  # 60 - 35 = 25
        assert zone_a[-1].step_index == 54
        assert zone_b[0].step_index == 55
        assert zone_b[-1].step_index == 59

    def test_small_turn_log_handling(self) -> None:
        """When total turns <= 5, all turns should be in Zone B and Zone A empty."""
        turns = [
            TurnRecord(step_index=i, source="USER_EXPLICIT", turn_type="USER_INPUT", status="DONE", content=f"Msg {i}")
            for i in range(3)
        ]
        zone_a, zone_b = apply_sliding_window(turns)
        assert len(zone_a) == 0
        assert len(zone_b) == 3

    def test_zone_a_telemetry_compression(self) -> None:
        """Zone A compression must strip code blocks and format 1-line telemetry string."""
        turn = TurnRecord(
            step_index=14,
            source="MODEL",
            turn_type="PLANNER_RESPONSE",
            status="DONE",
            content="Here is the implementation:\n```python\ndef dummy():\n    return 42\n```\nDone.",
            tool_calls=[{"name": "edit_file", "args": {"Path": "pcompile/config.py"}}],
        )
        line = compress_zone_a_turn(turn)
        assert line == "T14 [MODEL]: PLANNER_RESPONSE -> edit_file(config.py) (DONE)"
        assert "def dummy" not in line

    def test_zone_b_token_truncation_and_traceback_preservation(self) -> None:
        """Zone B must truncate body according to budget while preserving stack traces."""
        stack_trace = (
            "Traceback (most recent call last):\n"
            "  File 'test.py', line 10, in <module>\n"
            "    raise ValueError('Test Failure')\n"
            "ValueError: Test Failure"
        )
        verbose_text = "Important preamble context. " * 80 + "\n\n" + stack_trace

        turn = TurnRecord(
            step_index=58,
            source="SYSTEM",
            turn_type="TOOL_RESULT",
            status="ERROR",
            content=verbose_text,
        )

        # Mode REFINE (350 tokens ~= 1400 chars)
        formatted = format_zone_b_turn(turn, mode="REFINE", max_tokens=100)
        assert "### Turn 58 [SYSTEM / TOOL_RESULT] (ERROR)" in formatted
        assert "[...truncated...]" in formatted
        assert "ValueError: Test Failure" in formatted


class TestInvariantExtractor:
    """Tests for 5-turn invariant extractor capturing user constraints."""

    def test_extract_user_constraints_from_5_turns_ago(self) -> None:
        """User constraints from 5 turns ago must be captured in [PINNED_USER_INVARIANTS]."""
        turns: list[TurnRecord] = [
            TurnRecord(
                step_index=50,
                source="USER_EXPLICIT",
                turn_type="USER_INPUT",
                status="DONE",
                content=(
                    "Always maintain zero idle VRAM footprint.\n"
                    "Do not scaffold files belonging to Phases 2, 3, 4, or 5 yet.\n"
                    "Strictly keep memory usage below 350 MiB."
                ),
            ),
            TurnRecord(
                step_index=51,
                source="MODEL",
                turn_type="PLANNER_RESPONSE",
                status="DONE",
                content="Understood, I will respect these constraints.",
            ),
            TurnRecord(
                step_index=52,
                source="USER_EXPLICIT",
                turn_type="USER_INPUT",
                status="DONE",
                content="Proceed with running the tests.",
            ),
            TurnRecord(
                step_index=53,
                source="MODEL",
                turn_type="PLANNER_RESPONSE",
                status="DONE",
                content="All tests passed.",
            ),
            TurnRecord(
                step_index=54,
                source="USER_EXPLICIT",
                turn_type="USER_INPUT",
                status="DONE",
                content="Only modify files in the allowed working set.",
            ),
            TurnRecord(
                step_index=55,
                source="MODEL",
                turn_type="PLANNER_RESPONSE",
                status="DONE",
                content="Acknowledged.",
            ),
        ]

        invariants = extract_pinned_invariants(turns, lookback_user_turns=5)

        # Check that constraints are extracted
        assert any("zero idle vram footprint" in inv.lower() for inv in invariants)
        assert any("do not scaffold files" in inv.lower() for inv in invariants)
        assert any("only modify files in the allowed working set" in inv.lower() for inv in invariants)

    def test_invariants_deduplication(self) -> None:
        """Duplicate constraints across multiple turns must be deduplicated."""
        turns = [
            TurnRecord(
                step_index=1,
                source="USER_EXPLICIT",
                turn_type="USER_INPUT",
                status="DONE",
                content="Never modify production database directly.",
            ),
            TurnRecord(
                step_index=3,
                source="USER_EXPLICIT",
                turn_type="USER_INPUT",
                status="DONE",
                content="Never modify production database directly.",
            ),
        ]
        invariants = extract_pinned_invariants(turns)
        matches = [inv for inv in invariants if "never modify production database" in inv.lower()]
        assert len(matches) == 1


class TestActiveWorkingSetFileEvictor:
    """Tests for phase parsing, working set classification, and stale file eviction."""

    def test_parse_task_and_plan_phases(self) -> None:
        """Verify parsing of completed phases vs active phase and declared working sets."""
        task_text = """
# Task Tracker
## Phase 1: Core IO (phase-1-core-io)
- [x] Task 1 (`pyproject.toml`)
- [x] Task 2 (`pcompile/session_selector.py`)

## Phase 2: Harvester (phase-2-telemetry)
- [ ] Task 3 (`pcompile/harvester.py`)
- [ ] Task 4 (`pcompile/evictor.py`)

## Phase 3: Modes
- [ ] Task 5 (`pcompile/modes.py`)
"""
        plan_text = """
### Phase 1: Core IO
#### Allowed Working Set — Phase 1
`pyproject.toml`, `.gitignore`, `task.md`, `pcompile/__init__.py`, `pcompile/config.py`, `pcompile/session_selector.py`, `tests/conftest.py`, `tests/test_phase1_selector.py`.

### Phase 2: Harvester
#### Allowed Working Set — Phase 2
`pcompile/harvester.py`, `pcompile/evictor.py`, `tests/test_phase2_harvester.py`, `task.md`.
"""
        completed, active, task_files = parse_task_phases(task_text)
        assert completed == [1]
        assert active == 2
        assert "pcompile/session_selector.py" in task_files[1]
        assert "pcompile/harvester.py" in task_files[2]

        plan_files = parse_plan_phases(plan_text)
        assert 1 in plan_files
        assert "pcompile/session_selector.py" in plan_files[1]
        assert "pcompile/harvester.py" in plan_files[2]

    def test_parse_plan_phases_inline_allowed_working_set(self) -> None:
        """Verify parsing of inline Allowed_Working_Set: syntax on the same line after a colon."""
        inline_plan = """
### Phase 1: Core IO
- **Allowed_Working_Set:** `pyproject.toml`, `pcompile/session_selector.py`
- Other metadata here...

### Phase 2: Engine
- **Allowed_Working_Set:** `pcompile/harvester.py`, `pcompile/evictor.py`
"""
        plan_files = parse_plan_phases(inline_plan)
        assert 1 in plan_files
        assert 2 in plan_files
        assert plan_files[1] == ["pcompile/session_selector.py", "pyproject.toml"]
        assert plan_files[2] == ["pcompile/evictor.py", "pcompile/harvester.py"]

    def test_completed_files_evicted_while_active_files_retained(self) -> None:
        """Completed phase files must appear in [EVICT_FROM_WORKING_SET], while active phase files stay in [ACTIVE_PHASE_FILES_ONLY]."""
        accessed_files = {
            "pcompile/session_selector.py",  # Phase 1 file read earlier
            "tests/test_phase1_selector.py",  # Phase 1 file read earlier
            "unrelated_doc.txt",              # Read earlier, never modified
            "pcompile/harvester.py",         # Phase 2 active file
        }
        recently_modified = {
            "pcompile/harvester.py",
        }
        active_phase_files = {
            "pcompile/harvester.py",
            "pcompile/evictor.py",
            "tests/test_phase2_harvester.py",
            "task.md",
        }
        completed_phase_files = {
            "pcompile/session_selector.py",
            "tests/test_phase1_selector.py",
            "pyproject.toml",
        }

        evicted, active = classify_working_set_files(
            accessed_files=accessed_files,
            recently_modified=recently_modified,
            active_phase_files=active_phase_files,
            completed_phase_files=completed_phase_files,
        )

        # Verification of [EVICT_FROM_WORKING_SET]
        assert "pcompile/session_selector.py" in evicted
        assert "tests/test_phase1_selector.py" in evicted
        assert "unrelated_doc.txt" in evicted
        assert "pcompile/harvester.py" not in evicted

        # Verification of [ACTIVE_PHASE_FILES_ONLY]
        assert "pcompile/harvester.py" in active
        assert "pcompile/evictor.py" in active
        assert "task.md" in active
        assert "pcompile/session_selector.py" not in active

    def test_evict_working_set_context_end_to_end(self, tmp_path: Path) -> None:
        """End-to-end integration test with mock session and project directory."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Write task.md with Phase 1 done, Phase 2 active
        (project_dir / "task.md").write_text(
            "## Phase 1: Core\n- [x] Done (`pcompile/session_selector.py`)\n\n"
            "## Phase 2: Engine\n- [ ] In Progress (`pcompile/harvester.py`, `pcompile/evictor.py`)\n",
            encoding="utf-8",
        )
        (project_dir / "implementation_plan.md").write_text(
            "### Phase 1: Core\n#### Allowed Working Set — Phase 1\n`pcompile/session_selector.py`\n\n"
            "### Phase 2: Engine\n#### Allowed Working Set — Phase 2\n`pcompile/harvester.py`, `pcompile/evictor.py`\n",
            encoding="utf-8",
        )

        session_dir = tmp_path / "session"
        log_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"
        records = [
            {
                "step_index": 0,
                "source": "MODEL",
                "type": "TOOL_RESULT",
                "status": "DONE",
                "content": "Reading session selector",
                "tool_calls": [{"name": "read_file", "args": {"Path": "pcompile/session_selector.py"}}],
            },
            {
                "step_index": 1,
                "source": "MODEL",
                "type": "PLANNER_RESPONSE",
                "status": "DONE",
                "content": "Editing harvester",
                "tool_calls": [{"name": "edit_file", "args": {"Path": "pcompile/harvester.py"}}],
            },
        ]
        create_rich_session_log(log_file, records)

        result = evict_working_set_context(session_dir=session_dir, project_dir=project_dir)

        assert "pcompile/session_selector.py" in result.evicted_files
        assert "pcompile/harvester.py" in result.active_files
        assert result.active_phase_number == 2
        assert result.completed_phases == [1]

        rendered = result.render_block()
        assert "[EVICT_FROM_WORKING_SET]" in rendered
        assert "- pcompile/session_selector.py" in rendered
        assert "[ACTIVE_PHASE_FILES_ONLY]" in rendered
        assert "- pcompile/harvester.py" in rendered


class TestHarvestSessionTelemetryIntegration:
    """Integration test verifying complete CPU log harvesting pipeline."""

    def test_harvest_session_telemetry_pipeline(self, tmp_path: Path) -> None:
        """Verify harvest_session_telemetry output structure and rendering."""
        session_dir = tmp_path / "session_full"
        log_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"

        records: list[dict[str, Any]] = []
        # Create 40 turns
        for i in range(40):
            if i == 32:
                content = "Always verify with pytest before pushing."
            else:
                content = f"Turn step {i}"
            records.append({
                "step_index": i,
                "source": "USER_EXPLICIT" if i % 2 == 0 else "MODEL",
                "type": "USER_INPUT" if i % 2 == 0 else "PLANNER_RESPONSE",
                "status": "DONE",
                "content": content,
                "tool_calls": [{"name": "edit_file", "args": {"Path": f"module_{i}.py"}}] if i % 2 == 1 else [],
            })
        create_rich_session_log(log_file, records)

        result = harvest_session_telemetry(session_dir, mode="REFINE")

        assert result.total_turns == 40
        assert result.harvested_turns == 35
        assert len(result.zone_a_ledger) == 30
        assert len(result.zone_b_turns) == 5
        assert any("always verify with pytest" in inv.lower() for inv in result.pinned_invariants)

        rendered = result.render_prompt_context()
        assert "Compressed Execution Ledger (Zone A)" in rendered
        assert "Recent Turn Context (Zone B)" in rendered
        assert "[PINNED_USER_INVARIANTS]" in rendered
