"""Ephemeral 'Burst & Purge' CLI entry point, Ollama client (keep_alive=0), and clipboard pipe."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from pcompile.config import resolve_brain_dir
from pcompile.modes import CompiledPrompt, compile_prompt
from pcompile.session_selector import extract_session_title, select_session


def copy_to_clipboard(text: str) -> bool:
    """Pipe text to the system clipboard across Windows, WSL, and Linux."""
    # Windows & WSL: clip.exe
    if shutil.which("clip.exe") is not None:
        try:
            subprocess.run(
                ["clip.exe"],
                input=text.encode("utf-16le" if sys.platform == "win32" else "utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass

    if sys.platform == "win32" and shutil.which("clip") is not None:
        try:
            subprocess.run(
                ["clip"],
                input=text.encode("utf-16le"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass

    # Wayland: wl-copy
    if shutil.which("wl-copy") is not None:
        try:
            subprocess.run(
                ["wl-copy"],
                input=text.encode("utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass

    # X11: xclip or xsel
    if shutil.which("xclip") is not None:
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass

    if shutil.which("xsel") is not None:
        try:
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text.encode("utf-8"),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            pass

    return False


def query_ephemeral_ollama(
    compiled: CompiledPrompt,
    model: str = "pcompile",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: float = 30.0,
) -> str:
    """Send ephemeral generation payload to Ollama with keep_alive=0.

    Falls back to compiled prompt text if local Ollama daemon is unreachable.
    """
    payload = {
        "model": model,
        "prompt": compiled.prompt_text,
        "stream": False,
        "keep_alive": 0,  # Enforce strict 0 MB idle VRAM footprint
        "options": {
            "num_ctx": compiled.num_ctx,
            "num_predict": compiled.num_predict,
            "num_gpu": 99,
            "num_thread": 12,
            "temperature": compiled.temperature,
        },
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            res_body = response.read().decode("utf-8")
            data = json.loads(res_body)
            gen_text = data.get("response", "").strip()
            if gen_text:
                return gen_text
            return compiled.prompt_text
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
        sys.stderr.write(
            f"[OLLAMA OFFLINE] Local Ollama service unreachable ({err}). "
            f"Using deterministic CPU-compiled prompt.\n"
        )
        return compiled.prompt_text


def print_telemetry_footer(
    mode: str,
    session_title: str,
    num_ctx: int,
    active_count: int,
    evicted_count: int,
) -> None:
    """Display clean telemetry execution summary to stderr."""
    footer = (
        f"\n[Mode: {mode} | Session: {session_title[:30]} | Context: {num_ctx} | "
        f"Active Files: {active_count} | Evicted Files: {evicted_count} | VRAM Purged (keep_alive=0): YES]"
    )
    sys.stderr.write(footer + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for pcompile."""
    parser = argparse.ArgumentParser(
        prog="pcompile",
        description="Ephemeral Antigravity Prompt & Context Compiler (Zero-Idle VRAM Footprint).",
    )

    # Operational Modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-st",
        "--step",
        action="store_true",
        help="STEP mode: Fast ~120-word phase-transition directive (default prompt: 'proceed to next phase').",
    )
    mode_group.add_argument(
        "-r",
        "--refine",
        action="store_true",
        help="REFINE mode: ~300-word feature/bugfix prompt with dynamic edge cases (default mode).",
    )
    mode_group.add_argument(
        "-a",
        "--arch",
        action="store_true",
        help="ARCHITECT mode: ~700-word master blueprint compiler prompt with phased dependencies.",
    )

    # Session Selector
    parser.add_argument(
        "-s",
        "--select",
        action="store_true",
        help="Interactive session selector: Picker for top-10 recent Antigravity conversations.",
    )

    # Fast / Offline Flags
    parser.add_argument(
        "--dry-harvest",
        action="store_true",
        help="Run CPU harvester, evictor, and deterministic compiler without calling Ollama (<50 ms).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Bypass Ollama inference call and return deterministic compiled prompt.",
    )

    # Clipboard control
    parser.add_argument(
        "--no-clipboard",
        action="store_true",
        help="Do not pipe output to system clipboard.",
    )

    # Ollama settings
    parser.add_argument(
        "--model",
        type=str,
        default="pcompile",
        help="Ollama model name (default: pcompile).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama API endpoint (default: http://localhost:11434/api/generate).",
    )

    # Brain directory override
    parser.add_argument(
        "--brain-dir",
        type=str,
        default=None,
        help="Override Antigravity brain/ directory path.",
    )

    # User input prompt (positional)
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Raw user instruction or thought to compile into a drift-proof prompt.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution flow."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # 1. Determine Mode & Prompt String
    raw_prompt_text = " ".join(args.prompt).strip()

    if args.step:
        mode = "STEP"
        effective_prompt = raw_prompt_text or "Proceed to next phase"
    elif args.arch:
        mode = "ARCHITECT"
        effective_prompt = raw_prompt_text or "Architect complete system implementation plan"
    else:
        mode = "REFINE"
        effective_prompt = raw_prompt_text or "Refine codebase implementation and execute active phase"

    # 2. Select Session
    brain_path = Path(args.brain_dir) if args.brain_dir else None
    try:
        session_path = select_session(select=args.select, brain_dir=brain_path)
    except FileNotFoundError as err:
        sys.stderr.write(f"[ERROR] {err}\n")
        return 1

    session_title = extract_session_title(session_path, session_path.name)

    # 3. Compile Prompt via CPU Harvester & Evictor
    compiled = compile_prompt(
        mode=mode,
        session_dir=session_path,
        project_dir=Path.cwd(),
        raw_prompt=effective_prompt,
    )

    # 4. Ollama Generation or Offline CPU Output
    if args.dry_harvest or args.offline:
        final_output = compiled.prompt_text
    else:
        final_output = query_ephemeral_ollama(
            compiled=compiled,
            model=args.model,
            url=args.ollama_url,
        )

    # 5. Output to stdout
    sys.stdout.write(final_output + "\n")
    sys.stdout.flush()

    # 6. Cross-Platform Clipboard Pipe
    copied = False
    if not args.no_clipboard:
        copied = copy_to_clipboard(final_output)

    # 7. Print Telemetry Footer
    print_telemetry_footer(
        mode=compiled.mode,
        session_title=session_title,
        num_ctx=compiled.num_ctx,
        active_count=len(compiled.eviction_result.active_files),
        evicted_count=len(compiled.eviction_result.evicted_files),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
