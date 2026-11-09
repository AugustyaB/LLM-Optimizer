"""Deep Hugging Face Prompt & Solution Constraint Extractor.

Why: Inspects both instruction and reference solution to extract concrete, task-specific
technical edge cases (exception types, memory bounds, locks, type safety, sanitization)
so fine-tuning teaches the model to infer deep, non-obvious engineering constraints.
"""

from __future__ import annotations

import re
from typing import Any

# Patterns to extract concrete software engineering constraints from reference code
EXCEPTION_PATTERN = re.compile(r"""(?:except|raise)\s+([A-Z][a-zA-Z0-9_]*(?:Error|Exception|Interrupt)?)""")
ASYNC_PATTERN = re.compile(r"""\b(async\s+def|await\s+|asyncio\.|CancelledError|ThreadPoolExecutor)\b""")
MEMORY_BOUND_PATTERN = re.compile(r"""\b(generator|yield\s+|chunk_size|buffer_size|itertools|deque|read\(\d+\))\b""")
DB_LOCK_PATTERN = re.compile(r"""(?:\bcommit\s*\(|\brollback\s*\(|\btransaction\b|\bBEGIN\s+TRANSACTION|\bSELECT\s+.*?\s+FOR\s+UPDATE|\batomic\s*\()""", re.IGNORECASE)
INPUT_CHECK_PATTERN = re.compile(r"""\b(isinstance|len\([^\)]+\)\s*==\s*0|assert\s+|if not [a-zA-Z0-9_]+:)\b""")
TYPE_SAFETY_PATTERN = re.compile(r"""\b(Optional\[|Union\[|List\[|Dict\[|Tuple\[|Callable\[|@dataclass|final)\b""")


def extract_solution_constraints(instruction: str, code: str) -> list[str]:
    """Extract concrete, problem-specific technical constraints from reference solution."""
    constraints: list[str] = []
    combined = f"{instruction}\n{code}"

    # 1. Concrete Exception Handling
    exceptions = sorted(set(EXCEPTION_PATTERN.findall(code)))
    if exceptions:
        top_exc = ", ".join(exceptions[:3])
        constraints.append(f"Exception Safety: Handle explicit failure modes ({top_exc}) with non-destructive fallback.")

    # 2. Concurrency & Cancellation
    if ASYNC_PATTERN.search(code):
        constraints.append("Concurrency Guard: Propagate task cancellation tokens and avoid blocking the event loop.")

    # 3. Memory & Streaming Bounds
    if MEMORY_BOUND_PATTERN.search(code) or "O(1)" in combined or "stream" in instruction.lower():
        constraints.append("Memory Bound: Use chunked streaming/generators to keep auxiliary space complexity bounded.")

    # 4. Transactional & DB Locking
    if DB_LOCK_PATTERN.search(code) or "transaction" in instruction.lower():
        constraints.append("Transactional Isolation: Execute mutations inside atomic blocks with rollback protection.")

    # 5. Input Validation & Edge Bounds
    if INPUT_CHECK_PATTERN.search(code):
        constraints.append("Input Validation: Guard against empty collections, None references, and index out-of-bounds.")

    # 6. Type Invariants & Immutability
    if TYPE_SAFETY_PATTERN.search(code):
        constraints.append("Type Invariant: Enforce static typing and preserve immutable data representations.")

    # Fallback to ensure at least 2 high-signal constraints if code is brief
    if len(constraints) < 2:
        constraints.append("Algorithmic Efficiency: Guard against worst-case regression on boundary conditions.")
        constraints.append("Resource Lifecycle: Guarantee deterministic resource release and error propagation.")

    return constraints[:4]


def load_hf_pairs_with_constraints(count: int = 500) -> list[dict[str, Any]]:
    """Attempt to stream real instructions + solutions from Hugging Face and extract constraints."""
    records: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset  # type: ignore

        candidates = ("mlabonne/Evol-Instruct-Python-1k", "nickrosh/Evol-Instruct-Code-80k-v1")
        for repo_id in candidates:
            try:
                ds = load_dataset(repo_id, split="train", streaming=True)
                for item in ds:
                    inst = (item.get("instruction") or item.get("prompt") or item.get("text") or "").strip()
                    code = (item.get("output") or item.get("response") or item.get("solution") or "").strip()
                    if len(inst) > 25:
                        constraints = extract_solution_constraints(inst, code)
                        records.append({
                            "instruction": inst.replace("\r\n", " "),
                            "output_code": code,
                            "constraints": constraints,
                        })
                        if len(records) >= count:
                            return records
            except Exception:
                continue
    except Exception:
        pass
    return records
