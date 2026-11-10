"""Ephemeral 'Burst & Purge' CLI entry point, Ollama client (keep_alive=0), and clipboard pipe.

Why: Exposes zero-overhead CLI with <50ms CPU harvester, ephemeral Ollama inference,
hybrid sandwich locking, side-by-side comparison, and cross-platform clipboard export.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from pcompile.modes import CompiledPrompt, compile_prompt
from pcompile.sandwich import apply_hybrid_sandwich, validate_llm_output
from pcompile.session_selector import extract_session_title, select_session


def copy_to_clipboard(text: str) -> bool:
    """Pipe text to the system clipboard across Windows, WSL, and Linux."""
    cmds = []
    if shutil.which("clip.exe") is not None:
        cmds.append(["clip.exe"])
    elif sys.platform == "win32" and shutil.which("clip") is not None:
        cmds.append(["clip"])
    if shutil.which("wl-copy") is not None:
        cmds.append(["wl-copy"])
    if shutil.which("xclip") is not None:
        cmds.append(["xclip", "-selection", "clipboard"])
    if shutil.which("xsel") is not None:
        cmds.append(["xsel", "--clipboard", "--input"])

    enc = "utf-16le" if sys.platform == "win32" else "utf-8"
    for cmd in cmds:
        try:
            subprocess.run(cmd, input=text.encode(enc), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            continue
    return False


def query_ephemeral_ollama(
    compiled: CompiledPrompt,
    model: str = "pcompile",
    url: str = "http://localhost:11434/api/generate",
    timeout_s: float = 30.0,
) -> str:
    """Send ephemeral generation payload to Ollama with keep_alive=0 and apply hybrid sandwich."""
    prompt_val = getattr(compiled, "model_input_payload", None)
    if not isinstance(prompt_val, str) or not prompt_val:
        prompt_val = getattr(compiled, "prompt_text", "")
        if not isinstance(prompt_val, str):
            prompt_val = str(prompt_val)

    payload = {
        "model": model,
        "prompt": prompt_val,
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
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
            gen_text = data.get("response", "").strip()
            if gen_text:
                mode_val = "REFINE"
                try:
                    mode_val = str(compiled.mode)
                except Exception:
                    pass
                if hasattr(compiled, "eviction_result"):
                    if not validate_llm_output(gen_text, mode=mode_val):
                        sys.stderr.write("[FALLBACK: LLM output failed schema check; using verified CPU prompt]\n")
                        return getattr(compiled, "prompt_text", "")
                try:
                    return apply_hybrid_sandwich(gen_text, compiled)
                except Exception:
                    return gen_text
            return getattr(compiled, "prompt_text", "")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as err:
        sys.stderr.write(f"[OLLAMA OFFLINE] {err}. Using deterministic CPU-compiled prompt.\n")
        return getattr(compiled, "prompt_text", "")


def print_telemetry_footer(
    mode: str,
    session_title: str,
    num_ctx: int,
    active_count: int,
    evicted_count: int,
    turns: int = 0,
) -> None:
    """Display clean telemetry execution summary to stderr with high-turn warning."""
    footer = (
        f"\n[Mode: {mode} | Session: {session_title[:30]} | Context: {num_ctx} | "
        f"Active Files: {active_count} | Evicted Files: {evicted_count} | VRAM Purged (keep_alive=0): YES]"
    )
    sys.stderr.write(footer + "\n")
    if turns > 35:
        sys.stderr.write("[NOTICE] Session exceeds 35 turns. Consider running 'pcompile --handoff' to bootstrap a clean chat.\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser for pcompile."""
    parser = argparse.ArgumentParser(
        prog="pcompile", description="Ephemeral Antigravity Prompt & Context Compiler (Zero-Idle VRAM Footprint)."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-st", "--step", action="store_true", help="STEP mode: Fast phase-transition directive.")
    mode_group.add_argument("-r", "--refine", action="store_true", help="REFINE mode: Feature/bugfix prompt (default).")
    mode_group.add_argument("-a", "--arch", action="store_true", help="ARCHITECT mode: Master blueprint compiler.")
    mode_group.add_argument("-ho", "--handoff", action="store_true", help="HANDOFF mode: Clean bootstrap directive.")

    parser.add_argument("-s", "--select", action="store_true", help="Interactive session selector.")
    parser.add_argument("--dry-harvest", action="store_true", help="Run CPU harvester and evictor only (<50 ms).")
    parser.add_argument("--offline", action="store_true", help="Bypass Ollama inference call.")
    parser.add_argument("--force-llm", action="store_true", help="Force LLM generation in STEP mode (bypasses CPU fast-path).")
    parser.add_argument("--compare", action="store_true", help="Side-by-side comparison of CPU baseline vs Neural output.")
    parser.add_argument("--no-clipboard", action="store_true", help="Do not pipe output to system clipboard.")
    parser.add_argument("--model", type=str, default="pcompile", help="Ollama model name (default: pcompile).")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama endpoint.")
    parser.add_argument("--brain-dir", type=str, default=None, help="Override Antigravity brain/ directory path.")
    parser.add_argument("prompt", nargs="*", help="Raw user instruction or thought to compile.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution flow."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    raw_prompt_text = " ".join(args.prompt).strip()

    if args.step:
        mode, effective_prompt = "STEP", raw_prompt_text or "Proceed to next phase"
    elif args.arch:
        mode, effective_prompt = "ARCHITECT", raw_prompt_text or "Architect complete system implementation plan"
    elif args.handoff:
        mode, effective_prompt = "HANDOFF", raw_prompt_text or "Compile clean session handoff and bootstrap fresh context"
    else:
        mode, effective_prompt = "REFINE", raw_prompt_text or "Refine codebase implementation and execute active phase"

    brain_path = Path(args.brain_dir) if args.brain_dir else None
    try:
        session_path = select_session(select=args.select, brain_dir=brain_path)
    except FileNotFoundError as err:
        sys.stderr.write(f"[ERROR] {err}\n")
        return 1

    session_title = extract_session_title(session_path, session_path.name)
    compiled = compile_prompt(
        mode=mode, session_dir=session_path, project_dir=Path.cwd(), raw_prompt=effective_prompt
    )

    if args.compare:
        cpu_out = compiled.prompt_text
        neural_out = query_ephemeral_ollama(compiled=compiled, model=args.model, url=args.ollama_url)
        sep = "=" * 70
        cmp_view = (
            f"{sep}\n[1] DETERMINISTIC CPU BASELINE (--dry-harvest)\n{sep}\n"
            f"{cpu_out}\n\n"
            f"{sep}\n[2] TRAINED QWEN3-8B NEURAL OUTPUT (Hybrid Sandwich)\n{sep}\n"
            f"{neural_out}\n{sep}"
        )
        sys.stdout.write(cmp_view + "\n")
        sys.stdout.flush()
        if not args.no_clipboard:
            copy_to_clipboard(neural_out)
    else:
        is_cpu_fast_path = (mode == "STEP" and not args.force_llm) or args.dry_harvest or args.offline
        if is_cpu_fast_path:
            final_output = compiled.prompt_text
        else:
            final_output = query_ephemeral_ollama(compiled=compiled, model=args.model, url=args.ollama_url)

        sys.stdout.write(final_output + "\n")
        sys.stdout.flush()

        if not args.no_clipboard:
            copy_to_clipboard(final_output)

    print_telemetry_footer(
        mode=compiled.mode,
        session_title=session_title,
        num_ctx=compiled.num_ctx,
        active_count=len(compiled.eviction_result.active_files),
        evicted_count=len(compiled.eviction_result.evicted_files),
        turns=compiled.harvest_result.total_turns,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
