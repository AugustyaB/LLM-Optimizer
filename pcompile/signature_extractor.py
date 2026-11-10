"""Multi-Language Code Signature Extractor for Python, Java, TypeScript, and SQL.

Why: Extracts lightweight API surfaces (<5ms CPU) from active files so LLM prompt payloads
anchor on concrete method, interface, and table signatures without full-file bloat.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

JAVA_CLASS_PATTERN = re.compile(
    r"""^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:public|protected|private)?\s*(?:abstract|static|final)?\s*(class|interface|record|enum)\s+([A-Za-z0-9_<>]+)""",
    re.MULTILINE,
)
JAVA_METHOD_PATTERN = re.compile(
    r"""^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:public|protected|private)\s+(?:static\s+)?(?:final\s+)?([A-Za-z0-9_<>[\]]+)\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)""",
    re.MULTILINE,
)
TS_DECL_PATTERN = re.compile(
    r"""^\s*(?:export\s+)?(?:declare\s+)?(class|interface|type|enum)\s+([A-Za-z0-9_<>]+)""",
    re.MULTILINE,
)
TS_FUNC_PATTERN = re.compile(
    r"""^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z0-9_]+)\s*\(([^)]*)\)(?::\s*([^{;\n]+))?""",
    re.MULTILINE,
)
SQL_HEADER_PATTERN = re.compile(
    r"""^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:UNIQUE\s+)?(?:TABLE|INDEX|VIEW|TRIGGER)\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z0-9_."]+)""",
    re.IGNORECASE | re.MULTILINE,
)


def _format_ast_function(fn_node: ast.FunctionDef | ast.AsyncFunctionDef, prefix: str) -> str:
    """Format AST function node into clean Python signature."""
    async_prefix = "async " if isinstance(fn_node, ast.AsyncFunctionDef) else ""
    args: list[str] = []
    for arg in fn_node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)
    ret_str = f" -> {ast.unparse(fn_node.returns)}" if fn_node.returns else ""
    return f"{prefix}{async_prefix}def {fn_node.name}({', '.join(args)}){ret_str}"


def _extract_python_signatures(content: str, max_sig: int = 12) -> list[str]:
    """Parse Python code using AST to extract class and function type signatures."""
    signatures: list[str] = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return signatures

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            base_str = f"({', '.join(bases)})" if bases else ""
            signatures.append(f"class {node.name}{base_str}")
            for subnode in node.body:
                if isinstance(subnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    signatures.append(_format_ast_function(subnode, prefix="    "))
                    if len(signatures) >= max_sig:
                        return signatures
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures.append(_format_ast_function(node, prefix=""))
        if len(signatures) >= max_sig:
            break
    return signatures


def _extract_java_signatures(content: str, max_sig: int = 12) -> list[str]:
    """Extract Java class/interface declarations and method signatures."""
    signatures: list[str] = []
    for match in JAVA_CLASS_PATTERN.finditer(content):
        signatures.append(f"{match.group(1)} {match.group(2)}")
        if len(signatures) >= max_sig:
            return signatures
    for match in JAVA_METHOD_PATTERN.finditer(content):
        ret_type, name, params = match.group(1).strip(), match.group(2).strip(), re.sub(r"\s+", " ", match.group(3).strip())
        signatures.append(f"    {ret_type} {name}({params})")
        if len(signatures) >= max_sig:
            break
    return signatures


def _extract_ts_signatures(content: str, max_sig: int = 12) -> list[str]:
    """Extract TypeScript classes, interfaces, types, and function signatures."""
    signatures: list[str] = []
    for match in TS_DECL_PATTERN.finditer(content):
        signatures.append(f"{match.group(1)} {match.group(2)}")
        if len(signatures) >= max_sig:
            return signatures
    for match in TS_FUNC_PATTERN.finditer(content):
        name, params = match.group(1).strip(), re.sub(r"\s+", " ", match.group(2).strip())
        ret = f": {match.group(3).strip()}" if match.group(3) else ""
        signatures.append(f"function {name}({params}){ret}")
        if len(signatures) >= max_sig:
            break
    return signatures


def _extract_sql_signatures(content: str, max_sig: int = 12) -> list[str]:
    """Extract SQL CREATE TABLE / INDEX / VIEW definitions."""
    signatures: list[str] = []
    for match in SQL_HEADER_PATTERN.finditer(content):
        signatures.append(re.sub(r"\s+", " ", match.group(0).strip()))
        if len(signatures) >= max_sig:
            break
    return signatures


def extract_file_signatures(file_path: Path, max_sig: int = 12) -> list[str]:
    """Extract public API signatures from a supported source file."""
    if not file_path.is_file():
        return []
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    ext = file_path.suffix.lower()
    if ext == ".py":
        return _extract_python_signatures(content, max_sig=max_sig)
    if ext in (".java", ".kt"):
        return _extract_java_signatures(content, max_sig=max_sig)
    if ext in (".ts", ".tsx", ".js", ".jsx"):
        return _extract_ts_signatures(content, max_sig=max_sig)
    if ext == ".sql":
        return _extract_sql_signatures(content, max_sig=max_sig)
    return []


def extract_active_file_signatures(
    active_files: list[str],
    root_dir: Path | None = None,
    max_files: int = 5,
    max_sig_per_file: int = 12,
    project_dir: Path | None = None,
    recently_modified_files: list[str] | None = None,
) -> str:
    """Format up to 5 active files' signatures into a concise prompt block, prioritizing hot files."""
    base = project_dir or root_dir or Path.cwd()
    blocks: list[str] = ["[ACTIVE_FILE_SIGNATURES]"]
    inspected_count = 0

    ordered_files: list[str] = []
    seen = set()
    for f in (recently_modified_files or []) + active_files:
        if f and f not in seen:
            seen.add(f)
            ordered_files.append(f)

    for rel_path in ordered_files:
        full_path = base / rel_path
        if not full_path.is_file():
            continue
        sigs = extract_file_signatures(full_path, max_sig=max_sig_per_file)
        if sigs:
            blocks.append(f"# File: {rel_path}")
            for s in sigs:
                blocks.append(f"  {s}")
            inspected_count += 1
            if inspected_count >= max_files:
                break

    if inspected_count == 0:
        blocks.append("- (none available)")
    return "\n".join(blocks)
