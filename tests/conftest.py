"""Pytest fixtures for mock Antigravity brain directories and session telemetry."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest


def create_jsonl_log(log_path: Path, num_turns: int = 10, first_user_prompt: str = "Test prompt") -> None:
    """Helper to generate mock transcript.jsonl entries."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for idx in range(num_turns):
            if idx == 0:
                record: dict[str, Any] = {
                    "step_index": idx,
                    "source": "USER_EXPLICIT",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "content": first_user_prompt,
                }
            elif idx % 2 == 1:
                record = {
                    "step_index": idx,
                    "source": "MODEL",
                    "type": "PLANNER_RESPONSE",
                    "status": "DONE",
                    "content": f"Assistant response step {idx}",
                }
            else:
                record = {
                    "step_index": idx,
                    "source": "USER_EXPLICIT",
                    "type": "USER_INPUT",
                    "status": "DONE",
                    "content": f"User follow-up turn {idx}",
                }
            f.write(json.dumps(record) + "\n")


def create_rich_session_log(
    log_path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Helper to write custom turn records directly to transcript.jsonl."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def mock_brain_dir(tmp_path: Path) -> Path:
    """Fixture creating a mock Antigravity brain directory with realistic sessions."""
    brain = tmp_path / "brain"
    brain.mkdir(parents=True, exist_ok=True)

    base_time = time.time() - 10000

    # Session 1: Oldest - has task.md with checklist items
    s1_uuid = "11111111-1111-1111-1111-111111111111"
    s1_dir = brain / s1_uuid
    s1_dir.mkdir(parents=True)
    create_jsonl_log(
        s1_dir / ".system_generated" / "logs" / "transcript.jsonl",
        num_turns=12,
        first_user_prompt="Initialize Project Alpha",
    )
    (s1_dir / "task.md").write_text(
        "# Project Alpha Task Tracker\n\n- [x] Step 1\n- [x] Step 2\n- [ ] Step 3\n",
        encoding="utf-8",
    )
    t1 = base_time + 1000
    os.utime(s1_dir, (t1, t1))

    # Session 2: Middle - has implementation_plan.md only
    s2_uuid = "22222222-2222-2222-2222-222222222222"
    s2_dir = brain / s2_uuid
    s2_dir.mkdir(parents=True)
    create_jsonl_log(
        s2_dir / ".system_generated" / "logs" / "transcript.jsonl",
        num_turns=24,
        first_user_prompt="Build Database Pipeline",
    )
    (s2_dir / "implementation_plan.md").write_text(
        "# Master Database Pipeline Blueprint\n\nDetails here...",
        encoding="utf-8",
    )
    t2 = base_time + 2000
    os.utime(s2_dir, (t2, t2))

    # Session 3: Newest - has task.md and implementation_plan.md
    s3_uuid = "33333333-3333-3333-3333-333333333333"
    s3_dir = brain / s3_uuid
    s3_dir.mkdir(parents=True)
    create_jsonl_log(
        s3_dir / ".system_generated" / "logs" / "transcript.jsonl",
        num_turns=40,
        first_user_prompt="Run Ephemeral Compiler",
    )
    (s3_dir / "task.md").write_text(
        "# Ephemeral Antigravity Compiler\n\n- [x] Phase 1\n- [ ] Phase 2\n",
        encoding="utf-8",
    )
    (s3_dir / "implementation_plan.md").write_text(
        "# Master Implementation Plan\n\nFull spec...",
        encoding="utf-8",
    )
    t3 = base_time + 3000
    os.utime(s3_dir, (t3, t3))

    # Session 4: Minimal - no markdown files, user prompt in transcript
    s4_uuid = "44444444-4444-4444-4444-444444444444"
    s4_dir = brain / s4_uuid
    s4_dir.mkdir(parents=True)
    create_jsonl_log(
        s4_dir / ".system_generated" / "logs" / "transcript.jsonl",
        num_turns=3,
        first_user_prompt="Create quick script for log parsing",
    )
    t4 = base_time + 500
    os.utime(s4_dir, (t4, t4))

    return brain
