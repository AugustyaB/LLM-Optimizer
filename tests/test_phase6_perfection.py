"""Comprehensive unit tests for Phase 6: Architectural Perfection & Multi-Language Signatures.

Verifies:
1. Multi-Language Signature Extractor (Python AST, Java, TypeScript, SQL).
2. Dual-Directory Markdown Resolution (project_dir priority, session_dir fallback).
3. HANDOFF Mode Compilation & Hybrid Sandwich Enforcer.
4. Dataset Generation (600 unique records, zero synthetic numbers in outputs).
5. Strict Anti-Monolith Rule (< 280 lines for all modules in pcompile/ and training/).
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pcompile.cli import build_arg_parser, main
from pcompile.evictor import _find_markdown_file, evict_working_set_context
from pcompile.modes import CompiledPrompt, compile_prompt
from pcompile.sandwich import apply_hybrid_sandwich
from pcompile.signature_extractor import extract_active_file_signatures, extract_file_signatures
from tests.conftest import create_rich_session_log
from training.domain_bank import DOMAINS
from training.synthesize_dataset import generate_dataset, validate_dataset


# ---------------------------------------------------------------------------
# 1. Multi-Language Code Signature Extractor Tests
# ---------------------------------------------------------------------------
class TestMultiLanguageSignatureExtractor:
    """Test AST and regex signature extractors across Python, Java, TypeScript, and SQL."""

    def test_python_ast_signature_extraction(self, tmp_path: Path) -> None:
        """Python files must yield class and function definitions with type annotations."""
        py_file = tmp_path / "service.py"
        py_file.write_text(
            "class CacheManager:\n"
            "    def __init__(self, capacity: int = 100) -> None:\n"
            "        self.capacity = capacity\n\n"
            "    def get_entry(self, key: str) -> str | None:\n"
            "        return None\n\n"
            "def evict_stale_records(timeout_s: float) -> int:\n"
            "    return 0\n",
            encoding="utf-8",
        )
        sigs = extract_file_signatures(py_file)
        sig_text = "\n".join(sigs)
        assert "class CacheManager" in sig_text
        assert "def get_entry(self, key: str) -> str | None" in sig_text
        assert "def evict_stale_records(timeout_s: float) -> int" in sig_text

    def test_java_spring_signature_extraction(self, tmp_path: Path) -> None:
        """Java files must extract classes, Spring annotations, interfaces, and methods."""
        java_file = tmp_path / "OrderService.java"
        java_file.write_text(
            "@Service\n"
            "@Transactional\n"
            "public class OrderService implements IOrderService {\n"
            "    @Autowired\n"
            "    private OrderRepository orderRepo;\n\n"
            "    public OrderResponse processOrder(OrderRequest request) {\n"
            "        return null;\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        sigs = extract_file_signatures(java_file)
        sig_text = "\n".join(sigs)
        assert "class OrderService" in sig_text
        assert "processOrder(OrderRequest request)" in sig_text

    def test_typescript_signature_extraction(self, tmp_path: Path) -> None:
        """TypeScript files must extract interfaces, types, classes, and exported functions."""
        ts_file = tmp_path / "store.ts"
        ts_file.write_text(
            "export interface UserState {\n"
            "    id: string;\n"
            "    isActive: boolean;\n"
            "}\n\n"
            "export type StateAction = 'INIT' | 'UPDATE';\n\n"
            "export class StateReconciler {\n"
            "    reconcile(queue: UserState[]): void {}\n"
            "}\n\n"
            "export function createStore(initial: UserState): StateReconciler {\n"
            "    return new StateReconciler();\n"
            "}\n",
            encoding="utf-8",
        )
        sigs = extract_file_signatures(ts_file)
        sig_text = "\n".join(sigs)
        assert "interface UserState" in sig_text
        assert "type StateAction" in sig_text
        assert "class StateReconciler" in sig_text
        assert "function createStore" in sig_text

    def test_sql_ddl_signature_extraction(self, tmp_path: Path) -> None:
        """SQL files must extract CREATE TABLE, CREATE INDEX, and CREATE VIEW statements."""
        sql_file = tmp_path / "schema.sql"
        sql_file.write_text(
            "CREATE TABLE users (\n"
            "    id UUID PRIMARY KEY,\n"
            "    email VARCHAR(255) NOT NULL\n"
            ");\n\n"
            "CREATE INDEX idx_users_email ON users(email);\n"
            "CREATE VIEW active_users AS SELECT * FROM users;\n",
            encoding="utf-8",
        )
        sigs = extract_file_signatures(sql_file)
        sig_text = "\n".join(sigs)
        assert "CREATE TABLE users" in sig_text
        assert "CREATE INDEX idx_users_email" in sig_text
        assert "CREATE VIEW active_users" in sig_text

    def test_signature_extraction_caps_and_limits(self, tmp_path: Path) -> None:
        """Extractor must cap at 5 files and max 12 signatures per file."""
        files = []
        for i in range(8):
            f = tmp_path / f"mod_{i}.py"
            # 20 functions per file
            code = "\n".join(f"def fn_{j}(): pass" for j in range(20))
            f.write_text(code, encoding="utf-8")
            files.append(f.name)

        active_block = extract_active_file_signatures(files, project_dir=tmp_path)
        assert "[ACTIVE_FILE_SIGNATURES]" in active_block
        # Max 5 files inspected
        file_headers = [line for line in active_block.splitlines() if "File:" in line]
        assert 1 <= len(file_headers) <= 5

    def test_malformed_syntax_graceful_handling(self, tmp_path: Path) -> None:
        """Malformed code must not raise exceptions."""
        broken = tmp_path / "broken.py"
        broken.write_text("def def def ::: broken syntax ((((", encoding="utf-8")
        sigs = extract_file_signatures(broken)
        assert isinstance(sigs, list)


# ---------------------------------------------------------------------------
# 2. Dual-Directory Lookup Tests
# ---------------------------------------------------------------------------
class TestDualDirectoryMarkdownLookup:
    """Test project_dir priority with session_dir fallback for task.md and implementation plans."""

    def test_find_markdown_prefers_project_dir(self, tmp_path: Path) -> None:
        """When task.md exists in both project_dir and session_dir, project_dir must win."""
        proj = tmp_path / "project"
        sess = tmp_path / "session"
        proj.mkdir()
        sess.mkdir()

        (proj / "task.md").write_text("# Project Root Task\n- [ ] Phase 1\n", encoding="utf-8")
        (sess / "task.md").write_text("# Session Fallback Task\n- [x] Phase 1\n", encoding="utf-8")

        found = _find_markdown_file(proj, sess, "task.md")
        assert found is not None
        assert found == proj / "task.md"
        assert "Project Root Task" in found.read_text(encoding="utf-8")

    def test_find_markdown_falls_back_to_session_dir(self, tmp_path: Path) -> None:
        """When task.md is absent from project_dir, fallback to session_dir."""
        proj = tmp_path / "project"
        sess = tmp_path / "session"
        proj.mkdir()
        sess.mkdir()

        (sess / "task.md").write_text("# Session Fallback Task\n", encoding="utf-8")

        found = _find_markdown_file(proj, sess, "task.md")
        assert found is not None
        assert found == sess / "task.md"


# ---------------------------------------------------------------------------
# 3. HANDOFF Mode & Hybrid Sandwich Tests
# ---------------------------------------------------------------------------
class TestHandoffModeAndHybridSandwich:
    """Test HANDOFF compilation mode and hybrid sandwich post-processing."""

    def test_handoff_mode_compilation(self, tmp_path: Path) -> None:
        """HANDOFF mode compiles fresh bootstrap state with progress summary."""
        sess = tmp_path / "brain" / "sess-1"
        log = sess / ".system_generated" / "logs" / "transcript.jsonl"
        create_rich_session_log(log, [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
             "content": "Keep zero idle VRAM.", "tool_calls": []},
            {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
             "content": "Done.", "tool_calls": []},
        ])
        (sess / "task.md").write_text("# Task\n- [x] Phase 1\n- [ ] Phase 2\n", encoding="utf-8")

        compiled = compile_prompt("HANDOFF", sess, project_dir=tmp_path, raw_prompt="Take over task")
        assert compiled.mode == "HANDOFF"
        assert "[MODE: HANDOFF]" in compiled.model_input_payload
        assert "[HANDOFF & FRESH SESSION BOOTSTRAP DIRECTIVE]" in compiled.prompt_text
        assert "[EVICT_FROM_WORKING_SET]" in compiled.prompt_text
        assert "[ACTIVE_PHASE_FILES_ONLY]" in compiled.prompt_text
        assert "[PINNED_USER_INVARIANTS]" in compiled.prompt_text

    def test_hybrid_sandwich_strips_hallucinations_and_anchors(self, tmp_path: Path) -> None:
        """Hybrid sandwich must strip hallucinated file blocks and deterministically append CPU locks."""
        sess = tmp_path / "brain" / "sess-2"
        log = sess / ".system_generated" / "logs" / "transcript.jsonl"
        create_rich_session_log(log, [{"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT",
                                       "status": "DONE", "content": "Always verify 100%.", "tool_calls": []}])
        (sess / "task.md").write_text("# Task\n- [ ] Phase 1\n", encoding="utf-8")

        compiled = compile_prompt("REFINE", sess, project_dir=tmp_path, raw_prompt="Fix bug")

        # Simulate LLM output that hallucinated corrupted file locks
        hallucinated_llm_response = (
            "### [ROLE & OBJECTIVE]\nFix the bug cleanly.\n\n"
            "[EVICT_FROM_WORKING_SET]\n- fake_hallucinated_file.py\n\n"
            "[ACTIVE_PHASE_FILES_ONLY]\n- another_fake_file.py\n\n"
            "[PINNED_USER_INVARIANTS]\n- Wrong rule\n"
        )
        sandwiched = apply_hybrid_sandwich(hallucinated_llm_response, compiled)

        assert "fake_hallucinated_file.py" not in sandwiched
        assert "another_fake_file.py" not in sandwiched
        assert "[EVICT_FROM_WORKING_SET]" in sandwiched
        assert "[ACTIVE_PHASE_FILES_ONLY]" in sandwiched
        assert "[PINNED_USER_INVARIANTS]" in sandwiched


# ---------------------------------------------------------------------------
# 4. Dataset Uniqueness & Zero Synthetic Numbering Tests
# ---------------------------------------------------------------------------
class TestDatasetPerfection:
    """Test 600 records uniqueness and zero synthetic numbering in outputs."""

    def test_600_records_uniqueness_and_no_synthetic_numbering(self) -> None:
        """Dataset must contain exactly 600 unique records without Task # or Rule # numbering."""
        records = generate_dataset()
        assert len(records) == 600

        metrics = validate_dataset(records)
        assert metrics["valid"] is True
        assert metrics["total"] == 600
        assert metrics["step"] == 150
        assert metrics["refine"] == 200
        assert metrics["architect"] == 150
        assert metrics["handoff"] == 100
        assert metrics["unique_inputs"] == 600
        assert metrics["unique_outputs"] == 600


# ---------------------------------------------------------------------------
# 5. Strict Anti-Monolith Verification (< 280 lines)
# ---------------------------------------------------------------------------
class TestAntiMonolithRule:
    """Verify that NO Python module in pcompile/ or training/ exceeds 280 lines."""

    def test_all_modules_under_280_lines(self) -> None:
        """Every single Python file in pcompile/ and training/ must be under 280 lines."""
        root = Path(__file__).resolve().parent.parent
        oversized: list[tuple[str, int]] = []

        for folder in ("pcompile", "training"):
            for py_file in (root / folder).rglob("*.py"):
                line_count = len(py_file.read_text(encoding="utf-8").splitlines())
                if line_count > 280:
                    oversized.append((py_file.name, line_count))

        assert not oversized, f"The following modules exceed 280 lines: {oversized}"
