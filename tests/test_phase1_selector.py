"""Thorough unit tests for Phase 1: Session Selector & Cross-Platform Brain Resolver."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from pcompile.config import (
    ENV_BRAIN_DIR,
    IDLE_KEEP_ALIVE,
    MAX_TELEMETRY_WINDOW,
    MODE_CONFIGS,
    TARGET_VRAM_MB,
    VRAM_GUARD_THRESHOLD_MB,
    get_candidate_brain_paths,
    is_wsl,
    resolve_brain_dir,
)
from pcompile.session_selector import (
    SessionInfo,
    count_session_turns,
    extract_session_title,
    get_latest_session_dir,
    get_session_markdown_status,
    list_sessions,
    select_session,
    select_session_interactive,
)


class TestConfigAndBrainResolver:
    """Test path resolver, platform candidates, and configuration constants."""

    def test_hardware_invariants_and_token_budgets(self) -> None:
        """Verify hardware constants and mode token budgets match master plan."""
        assert TARGET_VRAM_MB == 8192
        assert VRAM_GUARD_THRESHOLD_MB == 350
        assert IDLE_KEEP_ALIVE == 0
        assert MAX_TELEMETRY_WINDOW == 35

        assert "STEP" in MODE_CONFIGS
        assert MODE_CONFIGS["STEP"].num_ctx == 2048
        assert MODE_CONFIGS["STEP"].num_predict == 250

        assert "REFINE" in MODE_CONFIGS
        assert MODE_CONFIGS["REFINE"].num_ctx == 4096
        assert MODE_CONFIGS["REFINE"].num_predict == 500

        assert "ARCHITECT" in MODE_CONFIGS
        assert MODE_CONFIGS["ARCHITECT"].num_ctx == 8192
        assert MODE_CONFIGS["ARCHITECT"].num_predict == 900

    def test_resolve_brain_dir_explicit_path(self, tmp_path: Path) -> None:
        """Explicit path argument must take highest priority."""
        explicit = tmp_path / "custom_brain"
        explicit.mkdir()
        resolved = resolve_brain_dir(explicit)
        assert resolved == explicit.resolve()

    def test_resolve_brain_dir_env_var(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """ANTIGRAVITY_BRAIN_DIR environment variable must override candidate paths."""
        env_brain = tmp_path / "env_brain"
        env_brain.mkdir()
        monkeypatch.setenv(ENV_BRAIN_DIR, str(env_brain))

        resolved = resolve_brain_dir()
        assert resolved == env_brain.resolve()

    def test_resolve_brain_dir_candidate_discovery(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolver must discover an existing candidate path when no env var is set."""
        monkeypatch.delenv(ENV_BRAIN_DIR, raising=False)
        fake_candidate = tmp_path / "discovered_brain"
        fake_candidate.mkdir()

        with patch("pcompile.config.get_candidate_brain_paths", return_value=[fake_candidate]):
            resolved = resolve_brain_dir()
            assert resolved == fake_candidate.resolve()

    def test_resolve_brain_dir_default_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Resolver must return canonical platform default if no candidate exists."""
        monkeypatch.delenv(ENV_BRAIN_DIR, raising=False)
        non_existent = [Path("/non/existent/path/brain")]

        with patch("pcompile.config.get_candidate_brain_paths", return_value=non_existent):
            resolved = resolve_brain_dir()
            assert isinstance(resolved, Path)
            assert "brain" in str(resolved)

    def test_candidate_brain_paths_non_empty(self) -> None:
        """Verify candidate brain paths list returns non-empty list on current platform."""
        candidates = get_candidate_brain_paths()
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_is_wsl_detection(self) -> None:
        """Verify WSL detection executes without unhandled exceptions."""
        result = is_wsl()
        assert isinstance(result, bool)


class TestSessionSelectorFastPath:
    """Test fast-path latest session discovery and under-15ms performance budget."""

    def test_get_latest_session_dir_returns_newest(self, mock_brain_dir: Path) -> None:
        """Must return the newest session directory (Session 3)."""
        latest = get_latest_session_dir(mock_brain_dir)
        assert latest.name == "33333333-3333-3333-3333-333333333333"

    def test_get_latest_session_dir_performance_under_15ms(self, mock_brain_dir: Path) -> None:
        """Execution must complete in well under 15 ms."""
        # Warm-up run
        _ = get_latest_session_dir(mock_brain_dir)

        # Measure 10 runs
        runs = 10
        start = time.perf_counter()
        for _ in range(runs):
            _ = get_latest_session_dir(mock_brain_dir)
        elapsed = (time.perf_counter() - start) / runs

        # Hard budget: strictly under 15 milliseconds (0.015 s)
        assert elapsed < 0.015, f"Execution took {elapsed * 1000:.2f} ms (budget: 15 ms)"

    def test_get_latest_session_dir_empty_raises(self, tmp_path: Path) -> None:
        """Must raise FileNotFoundError when brain directory is empty."""
        empty_brain = tmp_path / "empty_brain"
        empty_brain.mkdir()
        with pytest.raises(FileNotFoundError, match="No conversation sessions found"):
            get_latest_session_dir(empty_brain)

    def test_get_latest_session_dir_nonexistent_raises(self, tmp_path: Path) -> None:
        """Must raise FileNotFoundError when brain directory does not exist."""
        nonexistent = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_latest_session_dir(nonexistent)


class TestSessionMetadataExtraction:
    """Test turn counting, title extraction, and markdown status parsing."""

    def test_count_session_turns(self, mock_brain_dir: Path) -> None:
        """Turn count must accurately parse jsonl lines."""
        s1 = mock_brain_dir / "11111111-1111-1111-1111-111111111111"
        s2 = mock_brain_dir / "22222222-2222-2222-2222-222222222222"
        s3 = mock_brain_dir / "33333333-3333-3333-3333-333333333333"
        s4 = mock_brain_dir / "44444444-4444-4444-4444-444444444444"

        assert count_session_turns(s1) == 12
        assert count_session_turns(s2) == 24
        assert count_session_turns(s3) == 40
        assert count_session_turns(s4) == 3

    def test_count_session_turns_missing_logs(self, tmp_path: Path) -> None:
        """Empty session directory without logs must return 0 turns."""
        empty_session = tmp_path / "no_logs_uuid"
        empty_session.mkdir()
        assert count_session_turns(empty_session) == 0

    def test_extract_session_title(self, mock_brain_dir: Path) -> None:
        """Title must prioritize task.md, then implementation_plan.md, then first user prompt."""
        s1 = mock_brain_dir / "11111111-1111-1111-1111-111111111111"
        s2 = mock_brain_dir / "22222222-2222-2222-2222-222222222222"
        s3 = mock_brain_dir / "33333333-3333-3333-3333-333333333333"
        s4 = mock_brain_dir / "44444444-4444-4444-4444-444444444444"

        assert extract_session_title(s1, s1.name) == "Project Alpha Task Tracker"
        assert extract_session_title(s2, s2.name) == "Master Database Pipeline Blueprint"
        assert extract_session_title(s3, s3.name) == "Ephemeral Antigravity Compiler"
        assert "Create quick script for log parsing" in extract_session_title(s4, s4.name)

    def test_extract_session_title_fallback_to_id(self, tmp_path: Path) -> None:
        """When neither markdown nor transcript exists, fallback to Session <uuid>."""
        bare_uuid = "abcdef12-3456-7890-abcd-ef1234567890"
        bare_dir = tmp_path / bare_uuid
        bare_dir.mkdir()
        title = extract_session_title(bare_dir, bare_uuid)
        assert title == "Session abcdef12"

    def test_get_session_markdown_status(self, mock_brain_dir: Path) -> None:
        """Verify markdown status parsing for task checklist and plans."""
        s1 = mock_brain_dir / "11111111-1111-1111-1111-111111111111"
        s2 = mock_brain_dir / "22222222-2222-2222-2222-222222222222"
        s3 = mock_brain_dir / "33333333-3333-3333-3333-333333333333"
        s4 = mock_brain_dir / "44444444-4444-4444-4444-444444444444"

        assert get_session_markdown_status(s1) == "task.md (2/3 tasks)"
        assert get_session_markdown_status(s2) == "plan.md"
        assert get_session_markdown_status(s3) == "task.md (1/2 tasks) + plan.md"
        assert get_session_markdown_status(s4) == "None"


class TestInteractiveSessionSelection:
    """Test listing, sorting, prompt_toolkit/CLI picker, and session dispatch."""

    def test_list_sessions_sorted_newest_first(self, mock_brain_dir: Path) -> None:
        """Sessions must be sorted newest first."""
        sessions = list_sessions(mock_brain_dir)
        assert len(sessions) == 4

        # Ordered: Session 3 (t3), Session 2 (t2), Session 1 (t1), Session 4 (t4)
        assert sessions[0].session_id == "33333333-3333-3333-3333-333333333333"
        assert sessions[1].session_id == "22222222-2222-2222-2222-222222222222"
        assert sessions[2].session_id == "11111111-1111-1111-1111-111111111111"
        assert sessions[3].session_id == "44444444-4444-4444-4444-444444444444"

    def test_list_sessions_with_limit(self, mock_brain_dir: Path) -> None:
        """Limit parameter must restrict output to top N sessions."""
        sessions = list_sessions(mock_brain_dir, limit=2)
        assert len(sessions) == 2
        assert sessions[0].session_id == "33333333-3333-3333-3333-333333333333"
        assert sessions[1].session_id == "22222222-2222-2222-2222-222222222222"

    def test_session_info_format_line(self, mock_brain_dir: Path) -> None:
        """Format line must follow `[Index] Title | Turn Count | Last Modified | Markdown Status`."""
        sessions = list_sessions(mock_brain_dir, limit=1)
        line = sessions[0].format_line(1)
        assert line.startswith("[1] Ephemeral Antigravity Compiler | Turns: 40 | Modified:")
        assert "Status: task.md (1/2 tasks) + plan.md" in line

    def test_select_session_interactive_custom_choice(self, mock_brain_dir: Path) -> None:
        """Selecting '2' must return the 2nd session."""
        printed_lines: list[str] = []

        def mock_input(prompt: str) -> str:
            return "2"

        selected = select_session_interactive(
            brain_dir=mock_brain_dir,
            top_n=10,
            input_func=mock_input,
            print_func=printed_lines.append,
        )
        assert selected.session_id == "22222222-2222-2222-2222-222222222222"
        assert any("Antigravity Sessions" in line for line in printed_lines)

    def test_select_session_interactive_default_on_empty_enter(self, mock_brain_dir: Path) -> None:
        """Pressing Enter (empty answer) must return the 1st session (newest)."""

        def mock_input(prompt: str) -> str:
            return ""

        selected = select_session_interactive(
            brain_dir=mock_brain_dir,
            top_n=10,
            input_func=mock_input,
            print_func=lambda _: None,
        )
        assert selected.session_id == "33333333-3333-3333-3333-333333333333"

    def test_select_session_interactive_invalid_input_falls_back(self, mock_brain_dir: Path) -> None:
        """Invalid number or text must warn and return the newest session."""
        printed_lines: list[str] = []

        def mock_input(prompt: str) -> str:
            return "not-a-number"

        selected = select_session_interactive(
            brain_dir=mock_brain_dir,
            top_n=10,
            input_func=mock_input,
            print_func=printed_lines.append,
        )
        assert selected.session_id == "33333333-3333-3333-3333-333333333333"
        assert any("Invalid selection" in line for line in printed_lines)

    def test_select_session_interactive_empty_brain_raises(self, tmp_path: Path) -> None:
        """Interactive selection in an empty brain directory must raise FileNotFoundError."""
        empty_brain = tmp_path / "empty"
        empty_brain.mkdir()
        with pytest.raises(FileNotFoundError, match="No Antigravity sessions found"):
            select_session_interactive(brain_dir=empty_brain)

    def test_select_session_dispatch_default(self, mock_brain_dir: Path) -> None:
        """select_session(select=False) must return Path to newest session."""
        path = select_session(select=False, brain_dir=mock_brain_dir)
        assert path.name == "33333333-3333-3333-3333-333333333333"

    def test_select_session_dispatch_interactive(self, mock_brain_dir: Path) -> None:
        """select_session(select=True) must invoke interactive picker."""
        with patch(
            "pcompile.session_selector.select_session_interactive",
            return_value=SessionInfo(
                session_id="custom-uuid",
                path=mock_brain_dir / "33333333-3333-3333-3333-333333333333",
                last_modified=pytest.approx(0),  # type: ignore
                turn_count=5,
                title="Custom",
                markdown_status="None",
            ),
        ) as mock_interactive:
            path = select_session(select=True, brain_dir=mock_brain_dir)
            mock_interactive.assert_called_once_with(mock_brain_dir)
            assert path.name == "33333333-3333-3333-3333-333333333333"
