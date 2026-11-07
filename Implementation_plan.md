# Master Implementation Plan: Ephemeral Antigravity Prompt & Context Compiler (`pcompile`)

## 1. System Overview & Target Hardware Envelope
Build a local, zero-idle-footprint CLI application and QLoRA fine-tuning pipeline (`pcompile`) that reads local Google Antigravity session telemetry (`brain/<conversation_id>/`), prunes stale file context, pins recent user invariants, and compiles raw user thoughts into drift-proof prompts optimized for Gemini 3.8 Flash.

- **Target Hardware:** HP Omen 16 | AMD Ryzen 9 7940HS (8C/16T) | 16 GB DDR5 RAM | NVIDIA RTX 4070 Laptop GPU (8 GB GDDR6 VRAM) | AMD Radeon 780M iGPU | PCIe 4.0 NVMe SSD.
- **Idle Footprint Requirement:** 0 MB VRAM and 0 MB active model RAM when not executing (`"keep_alive": 0`).
- **Execution Budget:** CPU log harvesting must complete in under 50 ms using standard library + lightweight parsers. GPU inference must fit 100% inside 8 GB VRAM using a 4-bit quantized (`Q4_K_M`) 7B/8B model (`Qwen2.5-Coder-7B-Instruct` or `Llama-3.1-8B-Instruct`) with 8-bit quantized KV cache (`OLLAMA_KV_CACHE_TYPE=q8_0`).

---

## 2. Global Execution Rules for Gemini 3.8 Flash (MANDATORY INVARIANTS)
1. **Strict Phase-by-Phase Execution:** Execute ONLY ONE phase at a time. When a phase's exit criteria and `pytest` suite pass, update `task.md`, create the specified Git commit and tag, push to `origin`, and **HALT IMMEDIATELY**. Wait for the user's explicit prompt before starting the next phase.
2. **Working-Set Scope Lock:** During any phase, read and modify ONLY the files listed under that phase's `Allowed_Working_Set`. Never scan unrelated directories, `.venv`, `.git`, or raw model weight files.
3. **No Toy Stubs or Placeholders:** Every module must be production-ready, modular, type-annotated Python 3.11+ code with comprehensive unit test coverage.
4. **Zero Conversational Fluff:** Output concise architectural status updates and test results only.

---

## 3. Repository Directory Structure
```text
antigravity-prompt-compiler/
├── implementation_plan.md       # This master blueprint
├── task.md                      # Live phase checklist updated by agent
├── pyproject.toml               # Package config & CLI entry point (`pcompile`)
├── .gitignore                   # Ignores .venv/, *.gguf, outputs/, __pycache__/
├── pcompile/
│   ├── __init__.py
│   ├── config.py                # Hardware constants, paths, & mode token budgets
│   ├── session_selector.py      # Cross-platform Antigravity brain/ scanner & UI picker
│   ├── harvester.py             # 35-turn two-zone sliding window & 5-turn rule pinner
│   ├── evictor.py               # Active file working-set diffing & garbage collection
│   ├── modes.py                 # Prompt payload builders for STEP, REFINE, ARCHITECT
│   └── cli.py                   # Ephemeral Ollama client (keep_alive=0) & clipboard pipe
├── training/
│   ├── vram_guard.py            # Pre-flight nvidia-smi check (<300 MiB VRAM used)
│   ├── synthesize_dataset.py    # Generates/validates 500 JSONL pairs across 3 modes
│   ├── train_qlora.py           # Unsloth 4-bit QLoRA trainer + Q4_K_M GGUF exporter
│   └── Modelfile.template       # Ollama Modelfile configuration
└── tests/
    ├── conftest.py              # Mock Antigravity brain/ UUID folders & JSONL fixtures
    ├── test_phase1_selector.py
    ├── test_phase2_harvester.py
    ├── test_phase3_modes.py
    ├── test_phase4_training.py
    └── test_phase5_cli.py

## 4. Phased Implementation Specifications
PHASE 1: Scaffolding, Cross-Platform Path Resolver & Interactive Chat Selector
    Objective: Initialize the Python package and build the Antigravity session discovery engine and interactive terminal selector (--select / -s).

    Allowed_Working_Set: pyproject.toml, .gitignore, task.md, pcompile/__init__.py, pcompile/config.py, pcompile/session_selector.py, tests/conftest.py, tests/test_phase1_selector.py.

    Technical Requirements:

        pcompile/config.py: Auto-resolve the Antigravity brain/ directory across Linux (~/.gemini/antigravity-cli/brain/), WSL (/mnt/c/Users/<user>/.gemini/antigravity-cli/brain/), and Windows (C:\Users\<user>\.gemini\antigravity-cli\brain\), overridable via ANTIGRAVITY_BRAIN_DIR env var.

        pcompile/session_selector.py: Scan all conversation UUID directories, sort by last-modified timestamp (newest first), count turns in .system_generated/logs/*.jsonl, and extract the top header from task.md or implementation_plan.md.

        Default behavior (when --select is False): Return the most recently modified conversation directory automatically in under 15 ms.

        Interactive behavior (--select / -s): Render a clean prompt_toolkit (or fallback numbered CLI) selector showing the top 10 recent sessions with [Index] Title | Turn Count | Last Modified | Markdown Status.

    Phase 1 Exit Criteria & Git Gate:

        Run pytest tests/test_phase1_selector.py (must pass 100% using mock UUID fixtures).

        Update task.md, run git add . && git commit -m "feat(phase-1): session selector and cross-platform brain resolver" && git tag phase-1-complete, push to GitHub, and HALT.

PHASE 2: CPU Harvester, 35-Turn Two-Zone Sliding Window, File Evictor & 5-Turn Rule Pinner
    Objective: Build the CPU pre-processor that solves 5-prompt instruction amnesia and file context pollution without touching the GPU.

    Allowed_Working_Set: pcompile/harvester.py, pcompile/evictor.py, tests/test_phase2_harvester.py, task.md. (Evict session_selector.py from active editing).

    Technical Requirements:

        Hard 35-Turn Cutoff (MAX_WINDOW = 35): In harvester.py, parse .system_generated/logs/*.jsonl. Immediately discard any turn older than total_turns - 35.

        Two-Zone Sliding Window:

            Zone A (Turns N-35 to N-6): Strip all raw code blocks and file contents. Compress each turn into a single-line telemetry entry: T<step_index> [<source>]: <action_type> -> <target_file_or_summary> (<status>).

            Zone B (Turns N-5 to N): Preserve high-resolution turn text, compiler/runtime stack traces, and modified file signatures, capped at 350 tokens per turn in REFINE mode and 800 tokens per turn in ARCHITECT mode.

        Active Working-Set File Evictor (evictor.py):

            Extract all file paths accessed via read_file / view_file / edit_file across the last 15 turns.

            Parse task.md and implementation_plan.md to identify which phase is currently active and which phases are marked [x] (completed).

            Classify every previously read file into two strict lists:

                [EVICT_FROM_WORKING_SET]: Files belonging to completed phases or unrelated directories that were read but never modified.

                [ACTIVE_PHASE_FILES_ONLY]: Files explicitly tied to the current phase or modified in the last 2 turns.

        5-Turn User Invariant Extractor (Anti-Amnesia Pinner):

            Extract the raw user prompts from the last 5 to 8 user turns.

            Identify imperative constraints (e.g., "do not use...", "always keep...", "only modify...", architectural rules) and format them into a [PINNED_USER_INVARIANTS] block to be placed at the very bottom of every compiled prompt.

    Phase 2 Exit Criteria & Git Gate:

        Run pytest tests/test_phase2_harvester.py verifying: (a) a 60-turn log truncates strictly to 35 turns, (b) completed phase files appear in [EVICT_FROM_WORKING_SET], and (c) user rules from 5 turns ago are extracted into [PINNED_USER_INVARIANTS].

        Update task.md, commit with tag phase-2-complete, push to GitHub, and HALT.

PHASE 3: Operational Mode Router (STEP, REFINE, ARCHITECT) & Dataset Synthesis Engine
    Objective: Implement the three lifecycle modes and generate the 500-example training dataset (dataset.jsonl) for QLoRA fine-tuning.

    Allowed_Working_Set: pcompile/modes.py, training/synthesize_dataset.py, tests/test_phase3_modes.py, task.md. (Evict Phase 1 & 2 implementation files from active editing).

    Technical Requirements:

        pcompile/modes.py — Three Distinct Lifecycle Modes:

            STEP Mode (--step | num_ctx: 2048 | num_predict: 250):

                Use case: User just wants to say "proceed" to the next phase in implementation_plan.md.

                What it checks: Does NOT waste compute auditing code correctness. Inspects task.md vs implementation_plan.md to identify the next uncompleted phase, pulls [EVICT_FROM_WORKING_SET] and [ACTIVE_PHASE_FILES_ONLY] from evictor.py, and pulls [PINNED_USER_INVARIANTS] from the last 5 turns.

                Output format: A punchy ~120-word Phase-Transition Directive instructing Gemini 3.8 Flash to advance to Phase X, evict finished files, lock scope to Phase X files, and obey pinned rules.

            REFINE Mode (--refine | Default | num_ctx: 4096 | num_predict: 500):

                Use case: Continuous feature ideas, bug fixes, or mid-phase adjustments where intent alignment and unspoken coding edge cases matter.

                What it checks: Cross-references the user's raw prompt against implementation_plan.md, task.md, the 30-turn compressed ledger, and the last 5 turns.

                Output format: A structured ~300-word specification containing [ROLE & OBJECTIVE], [INTENT & ARCHITECTURE ALIGNMENT], [INFERRED CODING EDGE CASES] (memory streaming, concurrency/thread safety, null/error boundaries, non-destructive diffs), [CONTEXT SCOPE & FILE EVICTION LOCK], and bottom-anchored [PINNED_USER_INVARIANTS].

            ARCHITECT Mode (--arch | num_ctx: 8192 | num_predict: 900):

                Use case: Creating, auditing, or overhauling implementation_plan.md at project start or major pivots.

                What it checks: Full 35-turn audit, project file tree, and cross-phase dependency ordering.

                Output format: A comprehensive ~700-word Master Blueprint Compiler prompt that forces the target AI to produce a phased implementation_plan.md where every phase explicitly declares Allowed_Working_Set, Evict_On_Complete, Inferred_Edge_Cases, and Verification_Gate.

        training/synthesize_dataset.py:

            Build a deterministic + template/LLM-assisted generator and validator that produces training/dataset.jsonl containing 500 richly varied training pairs (150 STEP, 200 REFINE, 150 ARCHITECT) teaching the 7B/8B model how to infer unspoken software engineering edge cases and strict context-eviction blocks from harvested telemetry.

    Phase 3 Exit Criteria & Git Gate:

        Run pytest tests/test_phase3_modes.py verifying token budgets, mode formatting, and JSONL dataset schema integrity.

        Update task.md, commit with tag phase-3-complete, push to GitHub, and HALT.

PHASE 4: Hardware-Isolated Unsloth 4-Bit QLoRA Training & GGUF Export Pipeline
    Objective: Build the VRAM-safe QLoRA training script for the RTX 4070 Laptop GPU (8 GB VRAM) and automated Q4_K_M GGUF exporter.

    Allowed_Working_Set: training/vram_guard.py, training/train_qlora.py, training/Modelfile.template, tests/test_phase4_training.py, task.md.

    Technical Requirements:

        training/vram_guard.py: Query nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits. If used VRAM exceeds 350 MiB, print a clear warning identifying which processes are holding NVIDIA VRAM and instructing the user to offload them to the AMD Radeon 780M iGPU before training starts.

        training/train_qlora.py:

            Base model default: unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit (with CLI flag --base-model supporting unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit).

            Hyperparameters tuned for 8 GB GDDR6 + Ryzen 9 7940HS: max_seq_length=4096, load_in_4bit=True, r=16, lora_alpha=32, use_gradient_checkpointing="unsloth", per_device_train_batch_size=1, gradient_accumulation_steps=4, optim="adamw_8bit", bf16=True, dataset_num_proc=8, max_steps=350.

            Include a --smoke-test flag that validates dataset loading, ChatML prompt formatting, and configuration without running a 30-minute GPU loop during unit tests.

            Automatically export merged weights via model.save_pretrained_gguf("outputs/gguf", tokenizer, quantization_method="q4_k_m") and generate the Ollama Modelfile.

    Phase 4 Exit Criteria & Git Gate:

        Run pytest tests/test_phase4_training.py (testing vram_guard.py, dataset formatting, and --smoke-test configuration).

        Update task.md, commit with tag phase-4-complete (verifying .gitignore excludes large model binaries), push to GitHub, and HALT.

PHASE 5: Ephemeral "Burst & Purge" CLI (pcompile), KV Quantization & Clipboard Pipe
    Objective: Wire the complete end-to-end CLI utility so it sits at 0 MB idle memory, bursts onto the RTX 4070 on trigger, copies the compiled prompt to the system clipboard, and immediately purges VRAM back to 0 MB.

    Allowed_Working_Set: pcompile/cli.py, pyproject.toml, tests/test_phase5_cli.py, README.md, task.md.

    Technical Requirements:

        CLI Flags:

            pcompile --step (or -st): Runs STEP mode without requiring a prompt string (defaults raw input to "proceed to next phase").

            pcompile --refine "raw idea" (or -r, or default when no mode flag is given): Runs REFINE mode.

            pcompile --arch "project idea" (or -a): Runs ARCHITECT mode.

            --select (or -s): Opens the interactive Antigravity conversation picker before harvesting.

            --dry-harvest: Prints the CPU-harvested telemetry and file-eviction block without waking the GPU (for instant debugging).

        Ephemeral Ollama Execution:

            Pass "keep_alive": 0 in the /api/generate payload so Ollama unloads the model from VRAM and RAM the millisecond generation completes.

            Dynamically pass "options": {"num_ctx": mode_ctx, "num_predict": mode_predict, "num_gpu": 99, "num_thread": 12, "temperature": 0.2}.

            Ensure environment variables OLLAMA_KV_CACHE_TYPE=q8_0 and OLLAMA_FLASH_ATTENTION=1 are documented and applied in the service/launcher helper.

        Cross-Platform Clipboard Pipe:

            Automatically pipe the output to wl-copy (Wayland/Hyprland), xclip (X11), or clip.exe (Windows/WSL), and display a clean terminal summary showing: [Mode | Session Title | Harvested Tokens | Evicted Files Count | VRAM Purged: Yes].

    Phase 5 Exit Criteria & Git Gate:

        Run full test suite: pytest (100% pass across all 5 test modules).

        Update task.md and README.md with installation and usage commands, commit with tag phase-5-complete, push to GitHub, and HALT.