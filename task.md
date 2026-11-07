# Ephemeral Antigravity Prompt & Context Compiler (`pcompile`) - Task Tracker

## Phase 1: Scaffolding, Cross-Platform Path Resolver & Interactive Chat Selector (`phase-1-core-io`)
- [x] Initialize package configuration and project files (`pyproject.toml`, `.gitignore`)
- [x] Implement cross-platform Antigravity brain path auto-resolver (`pcompile/config.py`)
- [x] Implement session scanner, metadata extractor, and turn counter (`pcompile/session_selector.py`)
- [x] Implement fast default session selection (<15 ms) and interactive session selector (`prompt_toolkit` / CLI fallback)
- [x] Create pytest fixtures for mock brain UUID directories and JSONL logs (`tests/conftest.py`)
- [x] Write unit test suite and verify 100% pass rate (`tests/test_phase1_selector.py`)
- [x] Phase 1 verification gate & git commit/tag (`phase-1-complete`)

## Phase 2: CPU Harvester, 35-Turn Two-Zone Sliding Window, File Evictor & 5-Turn Rule Pinner (`phase-2-telemetry-engine`)
- [x] Implement JSONL log parser with hard 35-turn cutoff (`pcompile/harvester.py`)
- [x] Implement two-zone sliding window: Zone A compressed ledger and Zone B high-res context (`pcompile/harvester.py`)
- [x] Implement active working-set file evictor with `[EVICT_FROM_WORKING_SET]` and `[ACTIVE_PHASE_FILES_ONLY]` (`pcompile/evictor.py`)
- [x] Implement 5-turn user invariant extractor for anti-amnesia pinner (`[PINNED_USER_INVARIANTS]`)
- [x] Write unit tests for truncation, eviction, and invariant pinning (`tests/test_phase2_harvester.py`)
- [x] Phase 2 verification gate & git commit/tag (`phase-2-complete`)

## Phase 3: Operational Mode Router (STEP, REFINE, ARCHITECT) & Dataset Synthesis Engine (`phase-3-modes-dataset`)
- [x] Implement STEP mode prompt builder with phase transition directives (`pcompile/modes.py`)
- [x] Implement REFINE mode prompt builder with edge case inference and file eviction locks (`pcompile/modes.py`)
- [x] Implement ARCHITECT mode prompt builder for blueprint creation and overhaul (`pcompile/modes.py`)
- [x] Build dataset synthesis engine generating 500 validated training pairs in `dataset.jsonl` (`training/synthesize_dataset.py`)
- [x] Write unit tests for mode formatting, token budgets, and dataset schema (`tests/test_phase3_modes.py`)
- [x] Phase 3 verification gate & git commit/tag (`phase-3-complete`)

## Phase 4: Hardware-Isolated Unsloth 4-Bit QLoRA Training & GGUF Export Pipeline (`phase-4-qlora-pipeline`)
- [x] Implement VRAM guard checking nvidia-smi with threshold enforcement (`training/vram_guard.py`)
- [x] Implement Unsloth 4-bit QLoRA trainer with RTX 4070 hyperparameters and `--smoke-test` (`training/train_qlora.py`)
- [x] Implement automated Q4_K_M GGUF export and Ollama Modelfile generation (`training/Modelfile.template`)
- [x] Write unit tests for VRAM guard, dataset loading, and smoke test configuration (`tests/test_phase4_training.py`)
- [x] Phase 4 verification gate & git commit/tag (`phase-4-complete`)

## Phase 5: Ephemeral "Burst & Purge" CLI (`pcompile`), KV Quantization & Clipboard Pipe (`phase-5-ephemeral-cli`)
- [x] Implement ephemeral Ollama client with `keep_alive=0` and dynamic hardware options (`pcompile/cli.py`)
- [x] Implement CLI flags (`--step`, `--refine`, `--arch`, `--select`, `--dry-harvest`)
- [x] Implement cross-platform clipboard pipe (`wl-copy`, `xclip`, `clip.exe`)
- [x] Complete documentation (`README.md`) and package installation verification
- [x] Run complete test suite (`pytest`) across all phases
- [x] Phase 5 verification gate & git commit/tag (`phase-5-complete`)
