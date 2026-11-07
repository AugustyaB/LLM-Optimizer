"""CPU Telemetry Harvester with 35-turn two-zone sliding window and invariant pinner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcompile.config import MAX_TELEMETRY_WINDOW, RECENT_ZONE_WINDOW

ZONE_B_TOKEN_BUDGET: dict[str, int] = {"STEP": 250, "REFINE": 350, "ARCHITECT": 800, "HANDOFF": 600}

# Why: Regexes extract user imperative constraints across recent conversational turns.
INVARIANT_PATTERNS = [
    re.compile(r"\b(?:do not|don't|never|must not|cannot)\b[^.!?\n]+", re.IGNORECASE),
    re.compile(r"\b(?:always|must|strictly|ensure)\b[^.!?\n]+", re.IGNORECASE),
    re.compile(r"\b(?:only modify|touch only|create and touch only)\b[^.!?\n]+", re.IGNORECASE),
    re.compile(r"\b(?:working-set lock|zero-idle|keep alive|0 mb)\b[^.!?\n]+", re.IGNORECASE),
    re.compile(r"^[*-]\s*(?:do not|never|always|must|only modify|working-set)[^\n]+", re.IGNORECASE | re.MULTILINE),
]


@dataclass
class TurnRecord:
    """Structured representation of a single telemetry turn from transcript logs."""

    step_index: int
    source: str
    turn_type: str
    status: str
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw_record: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarvestResult:
    """Output of CPU log harvesting across two-zone sliding window."""

    total_turns: int
    harvested_turns: int
    zone_a_ledger: list[str]
    zone_b_turns: list[str]
    pinned_invariants: list[str]

    def render_prompt_context(self) -> str:
        """Render the complete harvested telemetry block for prompt compilation."""
        sections: list[str] = []
        if self.zone_a_ledger:
            sections.extend(["### Compressed Execution Ledger (Zone A)", "\n".join(self.zone_a_ledger)])
        if self.zone_b_turns:
            sections.extend(["### Recent Turn Context (Zone B)", "\n\n---\n\n".join(self.zone_b_turns)])
        if self.pinned_invariants:
            sections.extend(["[PINNED_USER_INVARIANTS]", *[f"- {inv}" for inv in self.pinned_invariants]])
        return "\n\n".join(sections)


def load_transcript_turns(session_dir: Path) -> list[TurnRecord]:
    """Parse .system_generated/logs/*.jsonl, preferring transcript.jsonl."""
    logs_dir = session_dir / ".system_generated" / "logs"
    if not logs_dir.is_dir():
        return []

    target_file = logs_dir / "transcript.jsonl"
    if not target_file.is_file():
        candidates = list(logs_dir.glob("*.jsonl"))
        if not candidates:
            return []
        target_file = candidates[0]

    turns: list[TurnRecord] = []
    try:
        with open(target_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    continue

                tool_calls = [tc for tc in data.get("tool_calls", []) if isinstance(tc, dict)]
                turns.append(
                    TurnRecord(
                        step_index=data.get("step_index", len(turns)),
                        source=data.get("source", "UNKNOWN"),
                        turn_type=data.get("type", "UNKNOWN"),
                        status=data.get("status", "DONE"),
                        content=data.get("content", "") or "",
                        tool_calls=tool_calls,
                        raw_record=data,
                    )
                )
    except Exception:
        return []
    return turns


def apply_sliding_window(
    turns: list[TurnRecord],
    max_window: int = MAX_TELEMETRY_WINDOW,
    recent_window: int = RECENT_ZONE_WINDOW,
) -> tuple[list[TurnRecord], list[TurnRecord]]:
    """Enforce hard 35-turn cutoff and divide into Zone A (compressed) and Zone B (high-res)."""
    if not turns:
        return [], []
    cutoff_turns = turns[-max_window:] if len(turns) > max_window else turns
    if len(cutoff_turns) <= recent_window:
        return [], cutoff_turns
    split_idx = len(cutoff_turns) - recent_window
    return cutoff_turns[:split_idx], cutoff_turns[split_idx:]


def _extract_turn_target_or_summary(turn: TurnRecord) -> str:
    """Extract a concise target file or summary action from turn data."""
    if turn.tool_calls:
        summaries: list[str] = []
        for tc in turn.tool_calls:
            name, args = tc.get("name", "tool"), tc.get("args", {}) or {}
            target = next((Path(str(args[k])).name for k in ("Path", "path", "file_path", "target_file") if k in args), "")
            if not target and "CommandLine" in args:
                target = str(args["CommandLine"]).strip().split()[0]
            summaries.append(f"{name}({target})" if target else name)
        return ", ".join(summaries[:3])

    content = turn.content.strip()
    if not content:
        return "<no content>"
    clean_line = re.sub(r"\s+", " ", re.sub(r"```.*?```", "[code]", content, flags=re.DOTALL).strip().splitlines()[0])
    return clean_line[:52] + "..." if len(clean_line) > 55 else clean_line or "<empty>"


def compress_zone_a_turn(turn: TurnRecord) -> str:
    """Compress Zone A turn into 1-line telemetry: T<step> [<source>]: <type> -> <target> (<status>)."""
    return f"T{turn.step_index} [{turn.source}]: {turn.turn_type} -> {_extract_turn_target_or_summary(turn)} ({turn.status})"


def format_zone_b_turn(turn: TurnRecord, mode: str = "REFINE", max_tokens: int | None = None) -> str:
    """Format Zone B turn preserving high-resolution text, traces, and file signatures."""
    char_budget = (max_tokens or ZONE_B_TOKEN_BUDGET.get(mode, 350)) * 4
    parts = [f"### Turn {turn.step_index} [{turn.source} / {turn.turn_type}] ({turn.status})"]

    if turn.tool_calls:
        parts.append("\n".join(
            f"Tool: `{tc.get('name')}` | Args: {json.dumps(tc.get('args', {}), ensure_ascii=False)[:117]}..."
            for tc in turn.tool_calls
        ))

    content = turn.content.strip()
    if content:
        if len(content) <= char_budget:
            parts.append(content)
        else:
            trace_match = re.search(r"(Traceback \(most recent call last\):[\s\S]+|FAILED [^\n]+[\s\S]+)", content)
            if trace_match:
                trace = trace_match.group(0).strip()
                rem = max(100, char_budget - len(trace) - 50)
                parts.append(f"{content[:rem].strip()}\n\n[...truncated...]\n\n{trace}")
            else:
                parts.append(content[:char_budget].strip() + "\n[...truncated...]")

    return "\n\n".join(parts)


def extract_pinned_invariants(turns: list[TurnRecord], lookback_user_turns: int = 8) -> list[str]:
    """Extract imperative user constraints from the last 5-8 user turns."""
    user_turns = [t for t in turns if t.source.upper() in {"USER_EXPLICIT", "USER"} or t.turn_type == "USER_INPUT"]
    invariants, seen = [], set()

    for turn in user_turns[-lookback_user_turns:]:
        for line in turn.content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            for pattern in INVARIANT_PATTERNS:
                for match in pattern.findall(stripped):
                    clean = re.sub(r"^[-*#\d.]+\s*", "", match.strip()).rstrip(".,;")
                    if len(clean) >= 10 and clean.lower() not in seen:
                        seen.add(clean.lower())
                        invariants.append(clean)
    return invariants


def harvest_session_telemetry(session_dir: Path, mode: str = "REFINE") -> HarvestResult:
    """Full CPU harvesting pipeline for an Antigravity session directory."""
    turns = load_transcript_turns(session_dir)
    zone_a_turns, zone_b_turns = apply_sliding_window(turns)
    harvested_window = zone_a_turns + zone_b_turns

    return HarvestResult(
        total_turns=len(turns),
        harvested_turns=len(harvested_window),
        zone_a_ledger=[compress_zone_a_turn(t) for t in zone_a_turns],
        zone_b_turns=[format_zone_b_turn(t, mode=mode) for t in zone_b_turns],
        pinned_invariants=extract_pinned_invariants(harvested_window),
    )
