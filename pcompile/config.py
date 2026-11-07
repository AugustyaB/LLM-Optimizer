"""Configuration, hardware constants, token budgets, and brain path resolver."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


# -------------------------------------------------------------------------
# Environment Variables & Discovery Constants
# -------------------------------------------------------------------------
ENV_BRAIN_DIR = "ANTIGRAVITY_BRAIN_DIR"

# -------------------------------------------------------------------------
# Hardware Envelope & Ephemeral Execution Invariants
# Target: HP Omen 16 (Ryzen 9 7940HS / RTX 4070 Laptop 8GB VRAM)
# -------------------------------------------------------------------------
TARGET_VRAM_MB: int = 8192
VRAM_GUARD_THRESHOLD_MB: int = 350
IDLE_KEEP_ALIVE: int = 0  # Force instant model offload from VRAM/RAM

OLLAMA_NUM_GPU: int = 99
OLLAMA_NUM_THREAD: int = 12
OLLAMA_TEMPERATURE: float = 0.2

# -------------------------------------------------------------------------
# Telemetry Window Invariants
# -------------------------------------------------------------------------
MAX_TELEMETRY_WINDOW: int = 35
RECENT_ZONE_WINDOW: int = 5
EVICTION_LOOKBACK_TURNS: int = 15


# -------------------------------------------------------------------------
# Operational Mode Specifications
# -------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeConfig:
    name: str
    num_ctx: int
    num_predict: int
    description: str


MODE_CONFIGS: dict[str, ModeConfig] = {
    "STEP": ModeConfig(
        name="STEP",
        num_ctx=2048,
        num_predict=250,
        description="Phase-transition director without redundant code verification.",
    ),
    "REFINE": ModeConfig(
        name="REFINE",
        num_ctx=4096,
        num_predict=500,
        description="Mid-phase prompt compiler with edge-case inference and invariant pinning.",
    ),
    "ARCHITECT": ModeConfig(
        name="ARCHITECT",
        num_ctx=8192,
        num_predict=900,
        description="Master blueprint compiler with phased dependencies and strict working sets.",
    ),
}


# -------------------------------------------------------------------------
# Cross-Platform Brain Path Resolver
# -------------------------------------------------------------------------
def is_wsl() -> bool:
    """Detect whether running inside Windows Subsystem for Linux (WSL)."""
    if sys.platform != "linux":
        return False
    if "WSL_DISTRO_NAME" in os.environ or "WSL_INTEROP" in os.environ:
        return True
    try:
        proc_version = Path("/proc/version")
        if proc_version.exists():
            return "microsoft" in proc_version.read_text(encoding="utf-8").lower()
    except Exception:
        pass
    return False


def get_candidate_brain_paths() -> list[Path]:
    """Return an ordered list of candidate Antigravity brain directories for current OS."""
    candidates: list[Path] = []
    home = Path.home()

    # 1. Standard home-relative candidates across platforms
    candidates.append(home / ".gemini" / "antigravity" / "brain")
    candidates.append(home / ".gemini" / "antigravity-cli" / "brain")

    # 2. Windows specific environment paths
    if sys.platform == "win32" or platform.system() == "Windows":
        userprofile = os.environ.get("USERPROFILE")
        if userprofile:
            u_path = Path(userprofile)
            candidates.append(u_path / ".gemini" / "antigravity" / "brain")
            candidates.append(u_path / ".gemini" / "antigravity-cli" / "brain")

        localappdata = os.environ.get("LOCALAPPDATA")
        if localappdata:
            l_path = Path(localappdata)
            candidates.append(l_path / "Google" / "Antigravity" / "brain")

    # 3. WSL environment: inspect Windows host drives (/mnt/c/Users/...)
    if is_wsl():
        mnt_c_users = Path("/mnt/c/Users")
        if mnt_c_users.exists() and mnt_c_users.is_dir():
            try:
                for user_entry in mnt_c_users.iterdir():
                    if user_entry.is_dir() and user_entry.name not in {"Public", "Default", "Default User", "All Users"}:
                        candidates.append(user_entry / ".gemini" / "antigravity" / "brain")
                        candidates.append(user_entry / ".gemini" / "antigravity-cli" / "brain")
            except Exception:
                pass

    # De-duplicate while preserving insertion order
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)

    return unique_candidates


def resolve_brain_dir(custom_path: Path | str | None = None) -> Path:
    """Resolve the Antigravity `brain/` directory.

    Order of resolution:
    1. Explicit custom_path argument (if supplied).
    2. ANTIGRAVITY_BRAIN_DIR environment variable (if non-empty).
    3. First existing candidate directory in standard platform locations.
    4. Canonical platform default path (even if not yet created).
    """
    if custom_path is not None:
        p = Path(custom_path).expanduser().resolve()
        return p

    env_val = os.environ.get(ENV_BRAIN_DIR, "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()

    candidates = get_candidate_brain_paths()
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    # Fallback default if no candidate exists yet
    if sys.platform == "win32" or platform.system() == "Windows":
        return (Path.home() / ".gemini" / "antigravity" / "brain").resolve()
    if is_wsl():
        return Path("/mnt/c/Users") / os.environ.get("USER", "user") / ".gemini" / "antigravity-cli" / "brain"
    return (Path.home() / ".gemini" / "antigravity-cli" / "brain").resolve()
