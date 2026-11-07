"""VRAM Guard: Pre-flight NVIDIA GPU memory verification and process offload diagnostics."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

from pcompile.config import TARGET_VRAM_MB, VRAM_GUARD_THRESHOLD_MB


@dataclass
class GPUProcessInfo:
    """Information regarding a process actively consuming GPU VRAM."""

    pid: int
    name: str
    used_mb: int


@dataclass
class VRAMStatus:
    """Diagnostic report of GPU VRAM status."""

    used_mb: int
    total_mb: int
    threshold_mb: int
    is_safe: bool
    processes: list[GPUProcessInfo] = field(default_factory=list)
    diagnostic_message: str = ""

    def format_diagnostic(self) -> str:
        """Format a human-readable diagnostic report."""
        if self.is_safe:
            return (
                f"[VRAM GUARD OK] GPU Memory Used: {self.used_mb} MiB / {self.total_mb} MiB "
                f"(Threshold: {self.threshold_mb} MiB). Safe to initiate QLoRA fine-tuning."
            )

        lines: list[str] = [
            f"[VRAM GUARD ALERT] GPU Memory Used: {self.used_mb} MiB exceeds threshold of {self.threshold_mb} MiB!",
            f"Target Hardware Envelope: RTX 4070 Laptop ({self.total_mb} MiB total GDDR6 VRAM).",
            "To prevent CUDA Out-Of-Memory (OOM) during 4-bit QLoRA training:",
            "Offload desktop applications, browsers, and background electron apps to the AMD Radeon 780M iGPU.",
        ]

        if self.processes:
            lines.append("\nActive Processes Holding VRAM:")
            for p in self.processes:
                lines.append(f"  - PID {p.pid}: {p.name} ({p.used_mb} MiB)")

        lines.append(
            "\nRemediation Steps:\n"
            "1. In Windows Settings -> System -> Display -> Graphics:\n"
            "   Set browsers (Opera, Chrome, Edge) and Electron apps to 'Power Saving (AMD Radeon 780M)'.\n"
            "2. Close unnecessary 3D applications or run train_qlora.py with --skip-vram-check if testing."
        )

        return "\n".join(lines)


def _default_cmd_runner(cmd: list[str]) -> str:
    """Execute a system command and return decoded standard output."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return res.stdout


def query_nvidia_vram(run_cmd: Callable[[list[str]], str] = _default_cmd_runner) -> tuple[int, int]:
    """Query current GPU used and total VRAM in MiB using nvidia-smi."""
    cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    out = run_cmd(cmd).strip()
    if not out:
        raise RuntimeError("Empty response from nvidia-smi memory query")

    # Handle multi-GPU or first line
    line = out.splitlines()[0].strip()
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Malformed nvidia-smi output: '{line}'")

    used_mb = int(parts[0])
    total_mb = int(parts[1])
    return used_mb, total_mb


def query_gpu_processes(run_cmd: Callable[[list[str]], str] = _default_cmd_runner) -> list[GPUProcessInfo]:
    """Query processes consuming GPU memory via nvidia-smi."""
    cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    processes: list[GPUProcessInfo] = []
    try:
        out = run_cmd(cmd).strip()
        if not out:
            return []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                pid = int(parts[0]) if parts[0].isdigit() else 0
                name = parts[1]
                used = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
                processes.append(GPUProcessInfo(pid=pid, name=name, used_mb=used))
    except Exception:
        pass
    return processes


def check_vram(
    threshold_mb: int = VRAM_GUARD_THRESHOLD_MB,
    enforce: bool = False,
    run_cmd: Callable[[list[str]], str] | None = None,
) -> VRAMStatus:
    """Check GPU VRAM against threshold and return diagnostic report.

    If `enforce` is True and VRAM exceeds threshold, raises RuntimeError.
    """
    cmd_fn = run_cmd or _default_cmd_runner

    # Check if nvidia-smi is installed on PATH
    if run_cmd is None and shutil.which("nvidia-smi") is None:
        status = VRAMStatus(
            used_mb=0,
            total_mb=TARGET_VRAM_MB,
            threshold_mb=threshold_mb,
            is_safe=True,
            diagnostic_message="nvidia-smi not found on PATH; VRAM guard running in simulated safe mode.",
        )
        return status

    try:
        used_mb, total_mb = query_nvidia_vram(run_cmd=cmd_fn)
        processes = query_gpu_processes(run_cmd=cmd_fn)
    except Exception as err:
        status = VRAMStatus(
            used_mb=0,
            total_mb=TARGET_VRAM_MB,
            threshold_mb=threshold_mb,
            is_safe=True,
            diagnostic_message=f"nvidia-smi query unavailable: {err}. Safe fallback applied.",
        )
        return status

    is_safe = used_mb <= threshold_mb
    status = VRAMStatus(
        used_mb=used_mb,
        total_mb=total_mb,
        threshold_mb=threshold_mb,
        is_safe=is_safe,
        processes=processes,
        diagnostic_message="",
    )
    status.diagnostic_message = status.format_diagnostic()

    if enforce and not is_safe:
        raise RuntimeError(status.diagnostic_message)

    return status
