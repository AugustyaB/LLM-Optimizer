"""Active working-set file evictor and context garbage collector.

Why: Dynamically identifies stale files from completed phases and earlier turns,
evicting them to prevent context drift while locking active phase components.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pcompile.config import EVICTION_LOOKBACK_TURNS
from pcompile.harvester import TurnRecord, load_transcript_turns

FILE_PATH_REGEX = re.compile(
    r"""(?:\b|['"`])([a-zA-Z0-9_\-./\\]+\.(?:py|toml|md|jsonl|json|txt|sh|yaml|yml|template|lock))(?:\b|['"`])"""
)
WRITE_TOOL_NAMES = {"edit_file", "write_file", "create_file", "apply_diff", "patch_file"}


@dataclass
class EvictionResult:
    """Result of working-set classification and stale context eviction."""

    active_files: list[str]
    evicted_files: list[str]
    active_phase_number: int | None
    completed_phases: list[int]
    recently_modified_files: list[str] = field(default_factory=list)

    def render_block(self) -> str:
        """Render eviction block for inclusion in prompt payloads."""
        evict = "\n".join(f"- {f}" for f in self.evicted_files) if self.evicted_files else "- (none)"
        act = "\n".join(f"- {f}" for f in self.active_files) if self.active_files else "- (none)"
        sections = [f"[EVICT_FROM_WORKING_SET]\n{evict}\n\n[ACTIVE_PHASE_FILES_ONLY]\n{act}"]
        hot = "\n".join(f"- {f}" for f in self.recently_modified_files) if self.recently_modified_files else "- (none)"
        sections.append(f"[RECENTLY_MODIFIED_HOT_FILES]\n{hot}")
        return "\n\n".join(sections)


def _normalize_path_str(raw_path: str, root_dir: Path | None = None) -> str:
    """Normalize file path to forward slashes and relative path where possible."""
    cleaned = raw_path.strip().strip("'\"`").replace("\\", "/")
    if root_dir is not None:
        try:
            p = Path(cleaned)
            if p.is_absolute():
                return p.resolve().relative_to(root_dir.resolve()).as_posix()
        except Exception:
            pass
    return re.sub(r"^\./", "", cleaned)


def parse_plan_phases(plan_content: str) -> dict[int, list[str]]:
    """Extract declared Allowed Working Set files for each phase from implementation_plan.md."""
    phase_files: dict[int, list[str]] = {}
    pattern = re.compile(r"Allowed[ _]Working[ _]Set[^:\n]*(?::\s*([^\n]+)|\n+([^\n#]+))", re.IGNORECASE)

    for block in re.split(r"(?=^###\s*Phase\s*\d+)", plan_content, flags=re.MULTILINE):
        phase_match = re.search(r"^###\s*Phase\s*(\d+)", block.strip(), flags=re.MULTILINE)
        if not phase_match:
            continue
        phase_num = int(phase_match.group(1))
        extracted: list[str] = []
        for match in pattern.finditer(block):
            raw_list = (match.group(1) or match.group(2) or "").strip()
            tokens = [_normalize_path_str(t) for t in re.findall(r"[`']?([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9_\-]+)[`']?", raw_list)]
            extracted.extend(tokens)
        if extracted:
            phase_files[phase_num] = sorted(set(extracted))
    return phase_files


def parse_task_phases(task_content: str) -> tuple[list[int], int | None, dict[int, list[str]]]:
    """Parse task.md to determine completed phases, active phase, and mentioned files."""
    completed_phases: list[int] = []
    active_phase: int | None = None
    phase_files: dict[int, list[str]] = {}

    for block in re.split(r"(?=^##\s*Phase\s*\d+)", task_content, flags=re.MULTILINE):
        stripped = block.strip()
        header = re.search(r"^##\s*Phase\s*(\d+)", stripped, flags=re.MULTILINE)
        if not header:
            continue
        phase_num = int(header.group(1))
        checkboxes = re.findall(r"^[*-]\s*\[([ xX])\]", stripped, flags=re.MULTILINE)
        if checkboxes:
            if all(cb.lower() == "x" for cb in checkboxes):
                completed_phases.append(phase_num)
            elif active_phase is None:
                active_phase = phase_num
        phase_files[phase_num] = sorted({_normalize_path_str(m) for m in FILE_PATH_REGEX.findall(stripped)})
    return completed_phases, active_phase, phase_files


def extract_turn_file_operations(
    turns: list[TurnRecord],
    lookback_turns: int = EVICTION_LOOKBACK_TURNS,
    root_dir: Path | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """Scan the last N turns for file accesses and mutations."""
    recent = turns[-lookback_turns:] if len(turns) > lookback_turns else turns
    accessed, modified, recently_mod = set(), set(), set()

    for idx, turn in enumerate(recent):
        is_recent_5 = (len(recent) - 1 - idx) < 5
        for tc in turn.tool_calls:
            t_name = tc.get("name", "").lower()
            args = tc.get("args", {}) or {}
            for k in ("Path", "path", "file_path", "target_file"):
                if k in args:
                    norm = _normalize_path_str(str(args[k]), root_dir)
                    if norm and not norm.startswith(".git"):
                        accessed.add(norm)
                        if any(w in t_name for w in WRITE_TOOL_NAMES):
                            modified.add(norm)
                            if is_recent_5:
                                recently_mod.add(norm)
            if "CommandLine" in args:
                for match in FILE_PATH_REGEX.findall(str(args["CommandLine"])):
                    norm = _normalize_path_str(match, root_dir)
                    if norm and not norm.startswith(".git"):
                        accessed.add(norm)
        for match in FILE_PATH_REGEX.findall(turn.content):
            norm = _normalize_path_str(match, root_dir)
            if norm and not norm.startswith(".git"):
                accessed.add(norm)
    return accessed, modified, recently_mod


def classify_working_set_files(
    accessed_files: set[str],
    recently_modified: set[str],
    active_phase_files: set[str],
    completed_phase_files: set[str],
) -> tuple[list[str], list[str]]:
    """Partition files into [EVICT_FROM_WORKING_SET] and [ACTIVE_PHASE_FILES_ONLY]."""
    active_set = set(active_phase_files) | set(recently_modified)
    evict_set = (set(completed_phase_files) | set(accessed_files)) - active_set
    clean_active = sorted({f for f in active_set if f and "." in f})
    clean_evict = sorted({f for f in evict_set if f and "." in f})
    return clean_evict, clean_active


def _find_markdown_file(
    project_dir: Path | None,
    session_dir: Path | None,
    filenames: str | list[str],
) -> Path | None:
    """Why: Checks project_dir first, then session_dir fallback for documentation."""
    names = [filenames] if isinstance(filenames, str) else list(filenames)
    candidates = [d for d in (project_dir, session_dir) if d is not None]
    for directory in candidates:
        for name in names:
            target = directory / name
            if target.is_file():
                return target
    return None


def evict_working_set_context(
    session_dir: Path,
    project_dir: Path | None = None,
    lookback_turns: int = EVICTION_LOOKBACK_TURNS,
) -> EvictionResult:
    """Analyze telemetry and project markdown files to evict stale context with dual-directory lookup."""
    root = project_dir or session_dir
    turns = load_transcript_turns(session_dir)

    task_path = _find_markdown_file(project_dir, session_dir, ["task.md"])
    task_content = task_path.read_text(encoding="utf-8", errors="replace") if task_path else ""

    plan_path = _find_markdown_file(project_dir, session_dir, ["implementation_plan.md", "Implementation_plan.md"])
    plan_content = plan_path.read_text(encoding="utf-8", errors="replace") if plan_path else ""

    completed_phases, active_phase, task_files = parse_task_phases(task_content)
    plan_files = parse_plan_phases(plan_content)

    completed_files: set[str] = set()
    for cp in completed_phases:
        completed_files.update(plan_files.get(cp, []))
        completed_files.update(task_files.get(cp, []))

    active_files: set[str] = set()
    if active_phase is not None:
        active_files.update(plan_files.get(active_phase, []))
        active_files.update(task_files.get(active_phase, []))

    accessed, _, recently_modified = extract_turn_file_operations(turns, lookback_turns=lookback_turns, root_dir=root)
    evicted_list, active_list = classify_working_set_files(accessed, recently_modified, active_files, completed_files)
    recently_mod_list = sorted({f for f in recently_modified if f and "." in f})

    return EvictionResult(
        active_files=active_list,
        evicted_files=evicted_list,
        active_phase_number=active_phase,
        completed_phases=completed_phases,
        recently_modified_files=recently_mod_list,
    )
