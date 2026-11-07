"""Antigravity session discovery engine, turn counter, and interactive selector."""

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
        mtime_str = self.last_modified.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[{index}] {self.title} | Turns: {self.turn_count} | "
            f"Modified: {mtime_str} | Status: {self.markdown_status}"
        )


def _count_lines_fast(file_path: Path) -> int:
    """Fast binary newline counting for JSONL logs without full text decoding."""
    try:
        with open(file_path, "rb") as f:
            count = 0
            buf_size = 1024 * 64
            while chunk := f.read(buf_size):
                count += chunk.count(b"\n")
            return count
    except Exception:
        return 0


def count_session_turns(session_dir: Path) -> int:
    """Count turns from `.system_generated/logs/*.jsonl`.

    Prefers `transcript.jsonl` if present; otherwise inspects the first matching *.jsonl file.
    """
    logs_dir = session_dir / ".system_generated" / "logs"
    if not logs_dir.is_dir():
        return 0

    transcript_file = logs_dir / "transcript.jsonl"
    if transcript_file.is_file():
        return _count_lines_fast(transcript_file)

    for entry in logs_dir.glob("*.jsonl"):
        if entry.is_file():
            return _count_lines_fast(entry)

    return 0


def _extract_header_from_file(file_path: Path) -> str | None:
    """Extract the first markdown header (# Header) from a markdown file."""
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
    """Extract session title from `task.md` or `implementation_plan.md` or logs."""
    # 1. Check task.md
    task_file = session_dir / "task.md"
    header = _extract_header_from_file(task_file)
    if header:
        return header

    # 2. Check implementation_plan.md (case-insensitive)
    for plan_name in ("implementation_plan.md", "Implementation_plan.md"):
        plan_file = session_dir / plan_name
        header = _extract_header_from_file(plan_file)
        if header:
            return header

    # 3. Fallback to first user input from transcript.jsonl
    transcript_file = session_dir / ".system_generated" / "logs" / "transcript.jsonl"
    if transcript_file.is_file():
        try:
            with open(transcript_file, "r", encoding="utf-8", errors="replace") as f:
                for _ in range(20):
                    line = f.readline()
                    if not line:
                        break
                    if '"USER_INPUT"' in line:
                        match = re.search(r'"content"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', line)
                        if match:
                            raw_content = match.group(1).replace("\\n", " ").strip()
                            clean_content = re.sub(r"\s+", " ", raw_content)
                            if clean_content:
                                return clean_content[:60] + ("..." if len(clean_content) > 60 else "")
        except Exception:
            pass

    return f"Session {session_id[:8]}" if len(session_id) >= 8 else f"Session {session_id}"


def get_session_markdown_status(session_dir: Path) -> str:
    """Determine markdown status from `task.md` and/or `implementation_plan.md`."""
    task_file = session_dir / "task.md"
    plan_files = [session_dir / "implementation_plan.md", session_dir / "Implementation_plan.md"]
    has_plan = any(p.is_file() for p in plan_files)

    if task_file.is_file():
        completed = 0
        total = 0
        try:
            with open(task_file, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("- [x]") or stripped.startswith("* [x]"):
                        completed += 1
                        total += 1
                    elif stripped.startswith("- [ ]") or stripped.startswith("* [ ]"):
                        total += 1
        except Exception:
            pass

        if total > 0:
            status_desc = f"task.md ({completed}/{total} tasks)"
        else:
            status_desc = "task.md"

        if has_plan:
            return f"{status_desc} + plan.md"
        return status_desc

    if has_plan:
        return "plan.md"

    return "None"


def get_latest_session_dir(brain_dir: Path | str | None = None) -> Path:
    """Return the most recently modified conversation directory in under 15 ms.

    Optimized for zero unnecessary file operations using `os.scandir`.
    """
    resolved_brain = resolve_brain_dir(brain_dir)
    if not resolved_brain.is_dir():
        raise FileNotFoundError(f"Antigravity brain directory does not exist: {resolved_brain}")

    latest_path: Path | None = None
    latest_mtime: float = -1.0

    try:
        with os.scandir(resolved_brain) as it:
            for entry in it:
                if entry.is_dir():
                    try:
                        mtime = entry.stat().st_mtime
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_path = Path(entry.path)
                    except OSError:
                        continue
    except OSError as err:
        raise FileNotFoundError(f"Cannot scan Antigravity brain directory: {resolved_brain}") from err

    if latest_path is None:
        raise FileNotFoundError(f"No conversation sessions found in {resolved_brain}")

    return latest_path


def list_sessions(
    brain_dir: Path | str | None = None,
    limit: int | None = None,
) -> list[SessionInfo]:
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

    # Sort descending by last modified timestamp
    entries.sort(key=lambda item: item[2], reverse=True)

    if limit is not None and limit > 0:
        entries = entries[:limit]

    results: list[SessionInfo] = []
    for session_id, path, mtime_val in entries:
        dt = datetime.fromtimestamp(mtime_val, tz=timezone.utc).astimezone()
        turns = count_session_turns(path)
        title = extract_session_title(path, session_id)
        status = get_session_markdown_status(path)

        results.append(
            SessionInfo(
                session_id=session_id,
                path=path,
                last_modified=dt,
                turn_count=turns,
                title=title,
                markdown_status=status,
            )
        )

    return results


def select_session_interactive(
    brain_dir: Path | str | None = None,
    top_n: int = 10,
    input_func: Callable[[str], str] | None = None,
    print_func: Callable[[str], None] | None = None,
) -> SessionInfo:
    """Render interactive session selector showing the top 10 sessions.

    Uses `prompt_toolkit` when available and connected to a TTY,
    otherwise falls back to a clean numbered CLI selector.
    """
    _print = print_func or print
    _input = input_func or input

    sessions = list_sessions(brain_dir=brain_dir, limit=top_n)
    if not sessions:
        resolved_brain = resolve_brain_dir(brain_dir)
        raise FileNotFoundError(f"No Antigravity sessions found in {resolved_brain}")

    _print("\nAntigravity Sessions (Most Recent First):")
    _print("-" * 88)
    for idx, s in enumerate(sessions, start=1):
        _print(s.format_line(idx))
    _print("-" * 88)

    prompt_label = f"Select session [1-{len(sessions)}] (default 1): "

    # Attempt prompt_toolkit prompt if no custom input_func supplied
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
        selected_idx = int(answer)
        if 1 <= selected_idx <= len(sessions):
            return sessions[selected_idx - 1]
    except ValueError:
        pass

    _print(f"Invalid selection '{answer}'. Defaulting to most recent session.")
    return sessions[0]


def select_session(select: bool = False, brain_dir: Path | str | None = None) -> Path:
    """Main entry point to obtain a session directory.

    If `select` is False, returns the newest session directory in under 15 ms.
    If `select` is True, opens the interactive session picker.
    """
    if not select:
        return get_latest_session_dir(brain_dir)
    selected = select_session_interactive(brain_dir)
    return selected.path
