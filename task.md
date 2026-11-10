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

## Phase 6: Architectural Perfection, Multi-Language Signatures & Handoff Mode (`phase-6-perfection`)
- [x] Multi-language code signature extractor for Python (AST), Java, TypeScript, and SQL (`pcompile/signature_extractor.py`)
- [x] Dual-directory markdown resolution prioritizing `project_dir` with `session_dir` fallback (`pcompile/evictor.py`)
- [x] HANDOFF mode (`--handoff` / `-ho`) for clean session bootstrap directives (`pcompile/config.py`, `pcompile/modes.py`)
- [x] Hybrid Sandwich compiler post-processor strictly enforcing CPU-verified file locks and invariants (`pcompile/sandwich.py`)
- [x] Anti-monolith refactor with dedicated domain bank (`training/domain_bank.py`, all modules < 280 lines)
- [x] Synthesize 600 unique records across STEP, REFINE, ARCHITECT, and HANDOFF modes with zero synthetic numbering in outputs (`training/synthesize_dataset.py`)
- [x] Upgrade base model to `unsloth/Qwen3-8B-unsloth-bnb-4bit` with `/no_think` suppression and response-only training loss (`training/train_qlora.py`, `training/Modelfile.template`)
- [x] Wire `--handoff`, high-turn warning (>35 turns), and hybrid sandwich into ephemeral CLI (`pcompile/cli.py`)
- [x] Write comprehensive unit test suite and verify 100% pass rate (`tests/test_phase6_perfection.py`)
- [x] Phase 6 verification gate & git commit/tag (`phase-6-complete`)

## Phase 7: Zero-Waste LLM Trimming, Anti-Overcorrection Guard, Hot-File Tracker & 4-Gate Evaluator (`phase-7-llm-verification`)
- [x] Zero-waste LLM output trimming (training outputs contain only generative sections; CPU sandwich appends deterministic blocks)
- [x] Instant CPU fast-path for STEP mode (`pcompile --step` runs <20 ms CPU bypass unless `--force-llm`)
- [x] Zone A consecutive turn deduplication (collapse consecutive matching turns into `... (xN)`)
- [x] Telemetry health signal (`CLEAN_PASSING` vs `ERROR_DETECTED`) and Anti-Overcorrection Guard preventing false-positive critiques
- [x] 20% clean-state Minimal Intervention Lock training pairs (`index % 5 == 0`)
- [x] Schema validation and fallback circuit breaker (`validate_llm_output`) in `pcompile/sandwich.py` and `pcompile/cli.py`
- [x] Log-driven hot-file tracker (`recently_modified_files`) and `[RECENTLY_MODIFIED_HOT_FILES]` block in `pcompile/evictor.py`
- [x] Active file signature prioritization for hot modified files in `pcompile/signature_extractor.py`
- [x] Zero-scan lock added to compiled prompts preventing broad directory scans
- [x] Deep Hugging Face prompt & solution constraint extractor (`training/hf_extractor.py`)
- [x] Ingest both instruction and reference code to infer task-specific constraints (exceptions, memory bounds, locks, type safety)
- [x] Force-regeneration of dataset on disk when `--use-hf` is enabled (`training/train_qlora.py`)
- [x] Leading `<think>...</think>` reasoning tag stripping in hybrid sandwich compiler (`pcompile/sandwich.py`)
- [x] Training metrics telemetry (`outputs/training_metrics.json`) recording initial/final loss, steps, and model metadata
- [x] Automated 4-gate model verification suite and scorecard CLI with clean-pass restraint audit (`training/evaluate_model.py`)
- [x] Side-by-side `--compare` mode in ephemeral CLI comparing CPU baseline against Neural output (`pcompile/cli.py`)
- [x] Comprehensive unit test suite with 107/107 passing tests (`tests/test_phase7_evaluation.py`)
- [x] Strict anti-monolith verification keeping all modules under 280 lines
- [x] Phase 7 verification gate & git commit/tag (`phase-7-complete`)
