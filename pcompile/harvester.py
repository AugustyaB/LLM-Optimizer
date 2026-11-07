"""CPU Telemetry Harvester with 35-turn two-zone sliding window and invariant pinner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pcompile.config import MAX_TELEMETRY_WINDOW, RECENT_ZONE_WINDOW

ZONE_B_TOKEN_BUDGET: dict[str, int] = {
    "STEP": 250,
    "REFINE": 350,
    "ARCHITECT": 800,
}

# Regex patterns for invariant constraint extraction
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
            sections.append("### Compressed Execution Ledger (Zone A)")
            sections.append("\n".join(self.zone_a_ledger))

        if self.zone_b_turns:
            sections.append("### Recent Turn Context (Zone B)")
            sections.append("\n\n---\n\n".join(self.zone_b_turns))

        if self.pinned_invariants:
            sections.append("[PINNED_USER_INVARIANTS]")
            for inv in self.pinned_invariants:
                sections.append(f"- {inv}")

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

                step_idx = data.get("step_index", len(turns))
                source = data.get("source", "UNKNOWN")
                turn_type = data.get("type", "UNKNOWN")
                status = data.get("status", "DONE")
                content = data.get("content", "") or ""

                tool_calls: list[dict[str, Any]] = []
                raw_tools = data.get("tool_calls", [])
                if isinstance(raw_tools, list):
                    for tc in raw_tools:
                        if isinstance(tc, dict):
                            tool_calls.append(tc)

                turns.append(
                    TurnRecord(
                        step_index=step_idx,
                        source=source,
                        turn_type=turn_type,
                        status=status,
                        content=content,
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
    """Enforce hard 35-turn cutoff and divide into Zone A and Zone B.

    - Hard cutoff: Discards any turn older than `total_turns - max_window`.
    - Zone A (Turns N-35 to N-6): Compressed ledger.
    - Zone B (Turns N-5 to N): High-resolution context.
    """
    total = len(turns)
    if total == 0:
        return [], []

    # Apply strict cutoff
    if total > max_window:
        cutoff_turns = turns[-max_window:]
    else:
        cutoff_turns = turns

    if len(cutoff_turns) <= recent_window:
        return [], cutoff_turns

    split_idx = len(cutoff_turns) - recent_window
    zone_a = cutoff_turns[:split_idx]
    zone_b = cutoff_turns[split_idx:]
    return zone_a, zone_b


def _extract_turn_target_or_summary(turn: TurnRecord) -> str:
    """Extract a concise target file or summary action from turn data."""
    # Check tool calls first
    if turn.tool_calls:
        summaries: list[str] = []
        for tc in turn.tool_calls:
            name = tc.get("name", "tool")
            args = tc.get("args", {}) or {}
            target = ""
            for key in ("Path", "path", "file_path", "target_file"):
                if key in args:
                    val = str(args[key])
                    # Get clean basename or short path
                    target = Path(val).name or val
                    break
            if not target and "CommandLine" in args:
                cmd = str(args["CommandLine"]).strip()
                target = cmd.split()[0] if cmd else "cmd"
            if target:
                summaries.append(f"{name}({target})")
            else:
                summaries.append(name)
        return ", ".join(summaries[:3])

    # If no tool calls, summarize text content
    content = turn.content.strip()
    if not content:
        return "<no content>"

    # Strip code blocks
    content_no_code = re.sub(r"```.*?```", "[code]", content, flags=re.DOTALL)
    first_line = content_no_code.strip().splitlines()[0].strip()
    clean_line = re.sub(r"\s+", " ", first_line)
    if len(clean_line) > 55:
        return clean_line[:52] + "..."
    return clean_line or "<empty>"


def compress_zone_a_turn(turn: TurnRecord) -> str:
    """Compress Zone A turn into a 1-line telemetry string.

    Format: `T<step_index> [<source>]: <type> -> <target_file_or_summary> (<status>)`
    """
    summary = _extract_turn_target_or_summary(turn)
    return f"T{turn.step_index} [{turn.source}]: {turn.turn_type} -> {summary} ({turn.status})"


def format_zone_b_turn(
    turn: TurnRecord,
    mode: str = "REFINE",
    max_tokens: int | None = None,
) -> str:
    """Format Zone B turn preserving high-resolution text, traces, and file signatures.

    Applies token budget truncation (350 tokens for REFINE, 800 tokens for ARCHITECT).
    Approximates 1 token ~= 4 characters.
    """
    budget_tokens = max_tokens or ZONE_B_TOKEN_BUDGET.get(mode, 350)
    char_budget = budget_tokens * 4

    header = f"### Turn {turn.step_index} [{turn.source} / {turn.turn_type}] ({turn.status})"
    parts: list[str] = [header]

    # Include tool call invocations
    if turn.tool_calls:
        tc_lines: list[str] = []
        for tc in turn.tool_calls:
            name = tc.get("name", "unknown_tool")
            args = tc.get("args", {})
            args_preview = json.dumps(args, ensure_ascii=False)
            if len(args_preview) > 120:
                args_preview = args_preview[:117] + "..."
            tc_lines.append(f"Tool: `{name}` | Args: {args_preview}")
        parts.append("\n".join(tc_lines))

    # Handle content and stack traces
    content = turn.content.strip()
    if content:
        if len(content) <= char_budget:
            parts.append(content)
        else:
            # Check for stack trace or test failure
            trace_match = re.search(
                r"(Traceback \(most recent call last\):[\s\S]+|FAILED [^\n]+[\s\S]+)",
                content,
            )
            if trace_match:
                trace = trace_match.group(0).strip()
                trace_len = len(trace)
                remaining_chars = max(100, char_budget - trace_len - 50)
                prefix = content[:remaining_chars].strip()
                truncated_body = f"{prefix}\n\n[...truncated...]\n\n{trace}"
            else:
                truncated_body = content[:char_budget].strip() + "\n[...truncated...]"
            parts.append(truncated_body)

    return "\n\n".join(parts)


def extract_pinned_invariants(
    turns: list[TurnRecord],
    lookback_user_turns: int = 8,
) -> list[str]:
    """Extract imperative user constraints from the last 5-8 user turns."""
    # Filter to user turns
    user_turns = [
        t for t in turns
        if t.source.upper() in {"USER_EXPLICIT", "USER"} or t.turn_type == "USER_INPUT"
    ]
    recent_user_turns = user_turns[-lookback_user_turns:]

    invariants: list[str] = []
    seen: set[str] = set()

    for turn in recent_user_turns:
        text = turn.content
        lines = text.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            for pattern in INVARIANT_PATTERNS:
                matches = pattern.findall(stripped)
                for match in matches:
                    clean = match.strip()
                    # Strip leading bullet dashes or asterisks
                    clean = re.sub(r"^[-*#\d.]+\s*", "", clean).strip()
                    # Remove trailing markdown punctuation
                    clean = clean.rstrip(".,;")
                    if len(clean) >= 10:
                        norm = clean.lower()
                        if norm not in seen:
                            seen.add(norm)
                            invariants.append(clean)

    return invariants


def harvest_session_telemetry(
    session_dir: Path,
    mode: str = "REFINE",
) -> HarvestResult:
    """Full CPU harvesting pipeline for an Antigravity session directory."""
    turns = load_transcript_turns(session_dir)
    total_turns = len(turns)

    zone_a_turns, zone_b_turns = apply_sliding_window(turns)

    # Process Zone A
    zone_a_ledger = [compress_zone_a_turn(t) for t in zone_a_turns]

    # Process Zone B
    zone_b_formatted = [format_zone_b_turn(t, mode=mode) for t in zone_b_turns]

    # Extract invariants across the harvested window
    harvested_window = zone_a_turns + zone_b_turns
    invariants = extract_pinned_invariants(harvested_window)

    return HarvestResult(
        total_turns=total_turns,
        harvested_turns=len(harvested_window),
        zone_a_ledger=zone_a_ledger,
        zone_b_turns=zone_b_formatted,
        pinned_invariants=invariants,
    )
