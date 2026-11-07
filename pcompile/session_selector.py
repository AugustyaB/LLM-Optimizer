"""Antigravity session discovery engine, turn counter, and interactive selector.

Why: Provides sub-15ms fast-path session resolution and fallback interactive picker
for recent Antigravity conversations across cross-platform environments.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pcompile.config import resolve_brain_dir


@dataclass
class SessionInfo:
    """Metadata for a discovered Antigravity session."""

    session_id: str
    path: Path
    last_modified: datetime
    turn_count: int
    title: str
    markdown_status: str

    def format_line(self, index: int) -> str:
        """Format entry as `[Index] Title | Turn Count | Last Modified | Markdown Status`."""
        mtime = self.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        return f"[{index}] {self.title} | Turns: {self.turn_count} | Modified: {mtime} | Status: {self.markdown_status}"


def _count_lines_fast(file_path: Path) -> int:
    """Fast binary newline counting for JSONL logs without full text decoding."""
    try:
        with open(file_path, "rb") as f:
            count, buf_size = 0, 1024 * 64
            while chunk := f.read(buf_size):
                count += chunk.count(b"\n")
            return count
    except Exception:
        return 0


def count_session_turns(session_dir: Path) -> int:
    """Count turns from `.system_generated/logs/*.jsonl`."""
    logs_dir = session_dir / ".system_generated" / "logs"
    if not logs_dir.is_dir():
        return 0
    t_file = logs_dir / "transcript.jsonl"
    if t_file.is_file():
        return _count_lines_fast(t_file)
    for entry in logs_dir.glob("*.jsonl"):
        if entry.is_file():
            return _count_lines_fast(entry)
    return 0


def _extract_header_from_file(file_path: Path) -> str | None:
    """Extract first markdown header from file."""
    if not file_path.is_file():
        return None
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(30):
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if stripped.startswith("#"):
                    clean = re.sub(r"^#+\s*", "", stripped).strip()
                    if clean:
                        return clean
    except Exception:
        return None
    return None


def extract_session_title(session_dir: Path, session_id: str) -> str:
    """Extract session title from task.md, implementation_plan.md, or logs."""
    for name in ("task.md", "implementation_plan.md", "Implementation_plan.md"):
        hdr = _extract_header_from_file(session_dir / name)
        if hdr:
            return hdr

    t_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"
    if t_file.is_file():
        try:
            with open(t_file, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    if '"USER_INPUT"' in line:
                        m = re.search(r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', line)
                        if m:
                            clean = re.sub(r"\s+", " ", m.group(1).replace("\\n", " ")).strip()
                            if clean:
                                return clean[:60] + ("..." if len(clean) > 60 else "")
        except Exception:
            pass
    return f"Session {session_id[:8]}" if len(session_id) >= 8 else f"Session {session_id}"


def get_session_markdown_status(session_dir: Path) -> str:
    """Determine markdown status from task.md and/or implementation_plan.md."""
    task_file = session_dir / "task.md"
    has_plan = any((session_dir / p).is_file() for p in ("implementation_plan.md", "Implementation_plan.md"))

    if task_file.is_file():
        completed, total = 0, 0
        try:
            with open(task_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("- [x]") or s.startswith("* [x]"):
                        completed += 1
                        total += 1
                    elif s.startswith("- [ ]") or s.startswith("* [ ]"):
                        total += 1
        except Exception:
            pass
        status = f"task.md ({completed}/{total} tasks)" if total > 0 else "task.md"
        return f"{status} + plan.md" if has_plan else status

    return "plan.md" if has_plan else "None"


def get_latest_session_dir(brain_dir: Path | str | None = None) -> Path:
    """Return the most recently modified conversation directory in under 15 ms."""
    resolved_brain = resolve_brain_dir(brain_dir)
    if not resolved_brain.is_dir():
        raise FileNotFoundError(f"Antigravity brain directory does not exist: {resolved_brain}")

    latest_path, latest_mtime = None, -1.0
    try:
        with os.scandir(resolved_brain) as it:
            for entry in it:
                if entry.is_dir():
                    try:
                        mtime = entry.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime, latest_path = mtime, Path(entry.path)
                    except OSError:
                        continue
    except OSError as err:
        raise FileNotFoundError(f"Cannot scan Antigravity brain directory: {resolved_brain}") from err

    if latest_path is None:
        raise FileNotFoundError(f"No conversation sessions found in {resolved_brain}")
    return latest_path


def list_sessions(brain_dir: Path | str | None = None, limit: int | None = None) -> list[SessionInfo]:
    """Scan all session directories, sort newest first, and extract metadata."""
    resolved_brain = resolve_brain_dir(brain_dir)
    if not resolved_brain.is_dir():
        return []

    entries: list[tuple[str, Path, float]] = []
    try:
        with os.scandir(resolved_brain) as it:
            for entry in it:
                if entry.is_dir():
                    try:
                        entries.append((entry.name, Path(entry.path), entry.stat().st_mtime))
                    except OSError:
                        continue
    except OSError:
        return []

    entries.sort(key=lambda item: item[2], reverse=True)
    if limit is not None and limit > 0:
        entries = entries[:limit]

    results: list[SessionInfo] = []
    for s_id, path, mtime in entries:
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc).astimezone()
        results.append(
            SessionInfo(
                session_id=s_id,
                path=path,
                last_modified=dt,
                turn_count=count_session_turns(path),
                title=extract_session_title(path, s_id),
                markdown_status=get_session_markdown_status(path),
            )
        )
    return results


def select_session_interactive(
    brain_dir: Path | str | None = None,
    top_n: int = 10,
    input_func: Callable[[str], str] | None = None,
    print_func: Callable[[str], None] | None = None,
) -> SessionInfo:
    """Render interactive session selector showing the top 10 sessions."""
    _print, _input = print_func or print, input_func or input
    sessions = list_sessions(brain_dir=brain_dir, limit=top_n)
    if not sessions:
        raise FileNotFoundError(f"No Antigravity sessions found in {resolve_brain_dir(brain_dir)}")

    _print("\nAntigravity Sessions (Most Recent First):")
    _print("-" * 88)
    for idx, s in enumerate(sessions, start=1):
        _print(s.format_line(idx))
    _print("-" * 88)

    prompt_label = f"Select session [1-{len(sessions)}] (default 1): "
    if input_func is None:
        try:
            from prompt_toolkit import prompt as pt_prompt
            answer = pt_prompt(prompt_label).strip()
        except Exception:
            answer = _input(prompt_label).strip()
    else:
        answer = _input(prompt_label).strip()

    if not answer:
        return sessions[0]
    try:
        idx = int(answer)
        if 1 <= idx <= len(sessions):
            return sessions[idx - 1]
    except ValueError:
        pass
    _print(f"Invalid selection '{answer}'. Defaulting to most recent session.")
    return sessions[0]


def select_session(select: bool = False, brain_dir: Path | str | None = None) -> Path:
    """Main entry point to obtain a session directory."""
    if not select:
        return get_latest_session_dir(brain_dir)
    return select_session_interactive(brain_dir).path
