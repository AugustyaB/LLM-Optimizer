"""Active working-set file evictor and context garbage collector."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pcompile.config import EVICTION_LOOKBACK_TURNS
from pcompile.harvester import TurnRecord, load_transcript_turns

# Regex to identify file paths with standard development extensions
FILE_PATH_REGEX = re.compile(
    r"""(?:\b|['"`])([a-zA-Z0-9_\-./\\]+\.(?:py|toml|md|jsonl|json|txt|sh|yaml|yml|template|lock))(?:\b|['"`])"""
)

READ_TOOL_NAMES = {"read_file", "view_file", "read_resource", "open_file", "grep"}
WRITE_TOOL_NAMES = {"edit_file", "write_file", "create_file", "apply_diff", "patch_file"}


@dataclass
class EvictionResult:
    """Result of working-set classification and stale context eviction."""

    active_files: list[str]
    evicted_files: list[str]
    active_phase_number: int | None
    completed_phases: list[int]

    def render_block(self) -> str:
        """Render eviction block for inclusion in prompt payloads."""
        lines: list[str] = ["[EVICT_FROM_WORKING_SET]"]
        if self.evicted_files:
            for f in self.evicted_files:
                lines.append(f"- {f}")
        else:
            lines.append("- (none)")

        lines.append("")
        lines.append("[ACTIVE_PHASE_FILES_ONLY]")
        if self.active_files:
            for f in self.active_files:
                lines.append(f"- {f}")
        else:
            lines.append("- (none)")

        return "\n".join(lines)


def _normalize_path_str(raw_path: str, root_dir: Path | None = None) -> str:
    """Normalize file path to forward slashes and relative path where possible."""
    cleaned = raw_path.strip().strip("'\"`").replace("\\", "/")
    if root_dir is not None:
        try:
            p = Path(cleaned)
            if p.is_absolute():
                rel = p.resolve().relative_to(root_dir.resolve())
                return rel.as_posix()
        except Exception:
            pass
    # Strip any leading ./
    return re.sub(r"^\./", "", cleaned)


def parse_plan_phases(plan_content: str) -> dict[int, list[str]]:
    """Extract declared Allowed Working Set files for each phase from implementation_plan.md.

    Matches both `Allowed Working Set` and `Allowed_Working_Set`, supporting both
    inline lists (e.g. `- **Allowed_Working_Set:** `f1.py`, `f2.py``) and
    multiline blocks following a heading.
    """
    phase_files: dict[int, list[str]] = {}
    phase_blocks = re.split(r"(?=^###\s*Phase\s*\d+)", plan_content, flags=re.MULTILINE)

    for block in phase_blocks:
        stripped = block.strip()
        phase_match = re.search(r"^###\s*Phase\s*(\d+)", stripped, flags=re.MULTILINE)
        if not phase_match:
            continue
        phase_num = int(phase_match.group(1))

        # Match Allowed Working Set or Allowed_Working_Set
        # Either inline after colon: `...Allowed[ _]Working[ _]Set[^:\n]*:\s*([^\n]+)`
        # Or following on next line: `...Allowed[ _]Working[ _]Set[^\n]*\n+([^\n#]+)`
        pattern = re.compile(
            r"Allowed[ _]Working[ _]Set[^:\n]*(?::\s*([^\n]+)|\n+([^\n#]+))",
            re.IGNORECASE,
        )
        extracted: list[str] = []
        for match in pattern.finditer(stripped):
            raw_list = (match.group(1) or match.group(2) or "").strip()
            tokens = [
                _normalize_path_str(token)
                for token in re.findall(r"[`']?([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9_\-]+)[`']?", raw_list)
            ]
            extracted.extend(tokens)

        if extracted:
            phase_files[phase_num] = sorted(set(extracted))

    return phase_files


def parse_task_phases(task_content: str) -> tuple[list[int], int | None, dict[int, list[str]]]:
    """Parse task.md to determine completed phases, active phase, and mentioned files.

    Returns:
        (completed_phase_numbers, active_phase_number, task_files_by_phase)
    """
    completed_phases: list[int] = []
    active_phase: int | None = None
    phase_files: dict[int, list[str]] = {}

    phase_blocks = re.split(r"(?=^##\s*Phase\s*\d+)", task_content, flags=re.MULTILINE)
    for block in phase_blocks:
        stripped = block.strip()
        if not stripped.startswith("## Phase"):
            continue

        header_match = re.search(r"^##\s*Phase\s*(\d+)", stripped, flags=re.MULTILINE)
        if not header_match:
            continue
        phase_num = int(header_match.group(1))

        # Check checkboxes
        checkboxes = re.findall(r"^[*-]\s*\[([ xX])\]", stripped, flags=re.MULTILINE)
        if checkboxes:
            is_all_completed = all(cb.lower() == "x" for cb in checkboxes)
            if is_all_completed:
                completed_phases.append(phase_num)
            elif active_phase is None:
                active_phase = phase_num

        # Extract file paths from phase block
        matches = FILE_PATH_REGEX.findall(stripped)
        extracted = [_normalize_path_str(m) for m in matches]
        phase_files[phase_num] = sorted(set(extracted))

    return completed_phases, active_phase, phase_files


def extract_turn_file_operations(
    turns: list[TurnRecord],
    lookback_turns: int = EVICTION_LOOKBACK_TURNS,
    root_dir: Path | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Scan the last N turns for file accesses.

    Returns:
        (accessed_files, modified_files, recently_modified_files)
    """
    recent = turns[-lookback_turns:] if len(turns) > lookback_turns else turns
    accessed_files: set[str] = set()
    modified_files: set[str] = set()
    recently_modified_files: set[str] = set()

    for idx, turn in enumerate(recent):
        turns_from_end = len(recent) - 1 - idx
        is_recent_2_turns = turns_from_end < 2

        # 1. Inspect structured tool calls
        for tc in turn.tool_calls:
            tool_name = tc.get("name", "").lower()
            args = tc.get("args", {}) or {}

            # Extract target file from arguments
            for key in ("Path", "path", "file_path", "target_file", "FilePath", "TargetFile"):
                if key in args:
                    raw_val = str(args[key])
                    norm = _normalize_path_str(raw_val, root_dir)
                    if norm and not norm.startswith(".git"):
                        accessed_files.add(norm)
                        if any(w in tool_name for w in WRITE_TOOL_NAMES):
                            modified_files.add(norm)
                            if is_recent_2_turns:
                                recently_modified_files.add(norm)

            if "CommandLine" in args:
                cmd_str = str(args["CommandLine"])
                for match in FILE_PATH_REGEX.findall(cmd_str):
                    norm = _normalize_path_str(match, root_dir)
                    if norm and not norm.startswith(".git"):
                        accessed_files.add(norm)

        # 2. Inspect text content
        for match in FILE_PATH_REGEX.findall(turn.content):
            norm = _normalize_path_str(match, root_dir)
            if norm and not norm.startswith(".git"):
                accessed_files.add(norm)

    return accessed_files, modified_files, recently_modified_files


def classify_working_set_files(
    accessed_files: set[str],
    recently_modified: set[str],
    active_phase_files: set[str],
    completed_phase_files: set[str],
) -> tuple[list[str], list[str]]:
    """Deterministically partition files into [ACTIVE_PHASE_FILES_ONLY] and [EVICT_FROM_WORKING_SET]."""
    active_set: set[str] = set()
    evict_set: set[str] = set()

    # Rule 1: Files explicitly in current active phase or modified within last 2 turns
    for f in active_phase_files:
        active_set.add(f)
    for f in recently_modified:
        active_set.add(f)

    # Rule 2: Files in completed phases (unless active in current phase)
    for f in completed_phase_files:
        if f not in active_set:
            evict_set.add(f)

    # Rule 3: Accessed files that were read in earlier turns, never modified, and not part of active phase
    for f in accessed_files:
        if f not in active_set:
            evict_set.add(f)

    # Ensure mutual exclusivity
    evict_set -= active_set

    # Filter out empty or invalid tokens
    clean_active = sorted({f for f in active_set if f and "." in f})
    clean_evict = sorted({f for f in evict_set if f and "." in f})

    return clean_evict, clean_active


def evict_working_set_context(
    session_dir: Path,
    project_dir: Path | None = None,
    lookback_turns: int = EVICTION_LOOKBACK_TURNS,
) -> EvictionResult:
    """Analyze session telemetry and project markdown files to evict stale context."""
    root = project_dir or session_dir
    turns = load_transcript_turns(session_dir)

    # 1. Parse phase states from task.md and implementation_plan.md
    task_file = root / "task.md"
    task_content = task_file.read_text(encoding="utf-8", errors="replace") if task_file.is_file() else ""

    plan_files = [root / "implementation_plan.md", root / "Implementation_plan.md"]
    plan_content = ""
    for pf in plan_files:
        if pf.is_file():
            plan_content = pf.read_text(encoding="utf-8", errors="replace")
            break

    completed_phases, active_phase, task_files_by_phase = parse_task_phases(task_content)
    plan_files_by_phase = parse_plan_phases(plan_content)

    # Combine file declarations
    completed_files: set[str] = set()
    for cp in completed_phases:
        completed_files.update(plan_files_by_phase.get(cp, []))
        completed_files.update(task_files_by_phase.get(cp, []))

    active_files: set[str] = set()
    if active_phase is not None:
        active_files.update(plan_files_by_phase.get(active_phase, []))
        active_files.update(task_files_by_phase.get(active_phase, []))

    # 2. Extract accessed and recently modified files from telemetry
    accessed, _, recently_modified = extract_turn_file_operations(
        turns,
        lookback_turns=lookback_turns,
        root_dir=root,
    )

    # 3. Classify into deterministic active and evicted sets
    evicted_list, active_list = classify_working_set_files(
        accessed_files=accessed,
        recently_modified=recently_modified,
        active_phase_files=active_files,
        completed_phase_files=completed_files,
    )

    return EvictionResult(
        active_files=active_list,
        evicted_files=evicted_list,
        active_phase_number=active_phase,
        completed_phases=completed_phases,
    )
