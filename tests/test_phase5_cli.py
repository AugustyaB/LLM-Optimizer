"""Thorough unit tests for Phase 5: Ephemeral CLI, Ollama client (keep_alive=0), and Clipboard Pipe."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from pcompile.cli import (
    build_arg_parser,
    copy_to_clipboard,
    main,
    print_telemetry_footer,
    query_ephemeral_ollama,
)
from pcompile.modes import CompiledPrompt
from tests.conftest import create_rich_session_log


@pytest.fixture
def mock_cli_session(tmp_path: Path) -> Path:
    """Fixture creating a minimal valid session for CLI integration tests."""
    brain = tmp_path / "brain"
    session_dir = brain / "55555555-5555-5555-5555-555555555555"
    log_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"

    records = [
        {
            "step_index": 0,
            "source": "USER_EXPLICIT",
            "type": "USER_INPUT",
            "status": "DONE",
            "content": "Always keep zero idle VRAM.",
            "tool_calls": [],
        },
        {
            "step_index": 1,
            "source": "MODEL",
            "type": "PLANNER_RESPONSE",
            "status": "DONE",
            "content": "Working on CLI implementation.",
            "tool_calls": [{"name": "write_file", "args": {"Path": "pcompile/cli.py"}}],
        },
    ]
    create_rich_session_log(log_file, records)

    (session_dir / "task.md").write_text(
        "# CLI Testing Task Tracker\n\n- [x] Phase 1\n- [ ] Phase 2\n",
        encoding="utf-8",
    )
    return brain


class TestCLIArgumentParser:
    """Test CLI argument parsing and mode flag resolution."""

    def test_default_mode_is_refine(self) -> None:
        """Invoking with a prompt string defaults to REFINE mode."""
        parser = build_arg_parser()
        args = parser.parse_args(["Optimize", "cache"])
        assert args.step is False
        assert args.arch is False
        assert args.prompt == ["Optimize", "cache"]

    def test_step_mode_flag(self) -> None:
        """--step flag sets step mode."""
        parser = build_arg_parser()
        args = parser.parse_args(["--step"])
        assert args.step is True
        assert args.arch is False

    def test_arch_mode_flag(self) -> None:
        """--arch flag sets architect mode."""
        parser = build_arg_parser()
        args = parser.parse_args(["-a", "Design", "blueprint"])
        assert args.arch is True
        assert args.step is False
        assert args.prompt == ["Design", "blueprint"]

    def test_dry_harvest_and_offline_flags(self) -> None:
        """--dry-harvest and --offline flags are parsed correctly."""
        parser = build_arg_parser()
        args = parser.parse_args(["--dry-harvest", "--offline", "--no-clipboard"])
        assert args.dry_harvest is True
        assert args.offline is True
        assert args.no_clipboard is True


class TestEphemeralOllamaClient:
    """Test Ollama client payload generation, keep_alive=0 enforcement, and offline fallback."""

    def test_ollama_payload_enforces_keep_alive_zero(self) -> None:
        """Ollama payload must strictly include 'keep_alive': 0 and mode token options."""
        compiled = MagicMock(spec=CompiledPrompt)
        compiled.prompt_text = "Test compiled prompt"
        compiled.num_ctx = 4096
        compiled.num_predict = 500
        compiled.temperature = 0.2

        captured_request: list[urllib.request.Request] = []

        def mock_urlopen(req: urllib.request.Request, timeout: float = 30.0) -> Any:
            captured_request.append(req)
            resp_data = json.dumps({"response": "Compiled output from Ollama"}).encode("utf-8")
            mock_resp = MagicMock()
            mock_resp.read.return_value = resp_data
            mock_resp.__enter__.return_value = mock_resp
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            res = query_ephemeral_ollama(compiled, model="pcompile-test", url="http://localhost:11434/api/generate")

        assert res == "Compiled output from Ollama"
        assert len(captured_request) == 1

        req = captured_request[0]
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "pcompile-test"
        assert payload["keep_alive"] == 0
        assert payload["stream"] is False
        assert payload["options"]["num_ctx"] == 4096
        assert payload["options"]["num_predict"] == 500
        assert payload["options"]["num_gpu"] == 99
        assert payload["options"]["num_thread"] == 12
        assert payload["options"]["temperature"] == 0.2

    def test_ollama_offline_fallback(self) -> None:
        """When Ollama is unreachable, gracefully fall back to CPU-compiled prompt text."""
        compiled = MagicMock(spec=CompiledPrompt)
        compiled.prompt_text = "CPU fallback prompt text"
        compiled.num_ctx = 2048
        compiled.num_predict = 250
        compiled.temperature = 0.2

        def mock_failing_urlopen(req: Any, timeout: float = 30.0) -> Any:
            raise urllib.error.URLError("Connection refused")

        with patch("urllib.request.urlopen", side_effect=mock_failing_urlopen):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                res = query_ephemeral_ollama(compiled, url="http://localhost:11434/api/generate")

        assert res == "CPU fallback prompt text"
        err_output = mock_err.getvalue()
        assert "[OLLAMA OFFLINE]" in err_output
        assert "Connection refused" in err_output


class TestClipboardPipe:
    """Test cross-platform clipboard piping across Windows, WSL, and Linux."""

    def test_copy_to_clipboard_windows_clip(self) -> None:
        """Test clip.exe clipboard pipe execution."""
        with patch("shutil.which", side_effect=lambda cmd: "/c/Windows/clip.exe" if "clip" in cmd else None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                success = copy_to_clipboard("test clipboard text")
                assert success is True
                mock_run.assert_called_once()

    def test_copy_to_clipboard_wayland_wl_copy(self) -> None:
        """Test wl-copy clipboard pipe execution."""
        with patch("shutil.which", side_effect=lambda cmd: "/usr/bin/wl-copy" if cmd == "wl-copy" else None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                success = copy_to_clipboard("wayland text")
                assert success is True
                assert mock_run.call_args[0][0] == ["wl-copy"]

    def test_copy_to_clipboard_x11_xclip(self) -> None:
        """Test xclip clipboard pipe execution."""
        with patch("shutil.which", side_effect=lambda cmd: "/usr/bin/xclip" if cmd == "xclip" else None):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                success = copy_to_clipboard("x11 text")
                assert success is True
                assert mock_run.call_args[0][0] == ["xclip", "-selection", "clipboard"]

    def test_copy_to_clipboard_failure_graceful(self) -> None:
        """When no clipboard tool is available, gracefully return False."""
        with patch("shutil.which", return_value=None):
            success = copy_to_clipboard("no tool")
            assert success is False


class TestTelemetryFooter:
    """Test telemetry execution summary formatting."""

    def test_print_telemetry_footer(self) -> None:
        """Verify footer renders mode, session, context, files, and VRAM purge status."""
        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
            print_telemetry_footer(
                mode="REFINE",
                session_title="Project Alpha Master Plan",
                num_ctx=4096,
                active_count=3,
                evicted_count=7,
            )
            output = mock_err.getvalue()

        assert "[Mode: REFINE" in output
        assert "Project Alpha Master Plan" in output
        assert "Context: 4096" in output
        assert "Active Files: 3" in output
        assert "Evicted Files: 7" in output
        assert "VRAM Purged (keep_alive=0): YES" in output


class TestEndToEndCLI:
    """End-to-end integration tests for pcompile CLI execution."""

    def test_cli_dry_harvest_refine(self, mock_cli_session: Path) -> None:
        """Verify pcompile --dry-harvest runs end-to-end and returns code 0."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                ret = main([
                    "--dry-harvest",
                    "--brain-dir",
                    str(mock_cli_session),
                    "--no-clipboard",
                    "Add buffered reader to parser",
                ])

        assert ret == 0
        stdout_val = mock_out.getvalue()
        stderr_val = mock_err.getvalue()

        assert "### [ROLE & OBJECTIVE]" in stdout_val
        assert "Add buffered reader to parser" in stdout_val
        assert "### [INFERRED CODING EDGE CASES]" in stdout_val
        assert "[EVICT_FROM_WORKING_SET]" in stdout_val
        assert "[ACTIVE_PHASE_FILES_ONLY]" in stdout_val
        assert "[Mode: REFINE" in stderr_val
        assert "VRAM Purged (keep_alive=0): YES" in stderr_val

    def test_cli_step_mode(self, mock_cli_session: Path) -> None:
        """Verify pcompile --step produces phase-transition directive."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            with patch("sys.stderr", new_callable=io.StringIO):
                ret = main([
                    "-st",
                    "--dry-harvest",
                    "--brain-dir",
                    str(mock_cli_session),
                    "--no-clipboard",
                ])

        assert ret == 0
        stdout_val = mock_out.getvalue()
        assert "### PHASE TRANSITION DIRECTIVE" in stdout_val

    def test_cli_arch_mode(self, mock_cli_session: Path) -> None:
        """Verify pcompile --arch produces master blueprint prompt."""
        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            with patch("sys.stderr", new_callable=io.StringIO):
                ret = main([
                    "-a",
                    "--dry-harvest",
                    "--brain-dir",
                    str(mock_cli_session),
                    "--no-clipboard",
                    "Architect modular telemetry engine",
                ])

        assert ret == 0
        stdout_val = mock_out.getvalue()
        assert "### [ROLE & ARCHITECTURAL MANDATE]" in stdout_val
        assert "Allowed_Working_Set" in stdout_val
