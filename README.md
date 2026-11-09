# `pcompile`: Ephemeral Antigravity Prompt & Context Compiler

A zero-idle-footprint prompt & context compiler, evaluation suite, and QLoRA fine-tuning pipeline designed for local developer pairs working with Google Antigravity and Gemini 3.8 Flash.

`pcompile` reads local Google Antigravity session telemetry (`brain/<conversation_id>/`), prunes stale file context, pins recent user invariants, extracts multi-language code signatures, infers deep technical coding constraints, and compiles raw thoughts into drift-proof prompt specifications.

---

## Target Hardware Envelope

Designed and tuned specifically for laptop dual-GPU architectures (such as the **HP Omen 16**):
- **CPU:** AMD Ryzen 9 7940HS (8 Cores / 16 Threads)
- **RAM:** 16 GB DDR5
- **Dedicated GPU (dGPU):** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6 VRAM)
- **Integrated GPU (iGPU):** AMD Radeon 780M (Shared System Memory)
- **Storage:** PCIe 4.0 NVMe SSD

### Zero-Idle Footprint & Hardware Isolation
- **Idle Memory:** Strictly 0 MB VRAM and 0 MB RAM when not actively compiling.
- **Offload Desktop Apps to iGPU:** In Windows Graphics Settings, route desktop apps, browsers (Opera, Chrome, Edge), and Electron tools to the **AMD Radeon 780M iGPU** so the full 8 GB VRAM on the RTX 4070 remains free for ephemeral inference and QLoRA fine-tuning.
- **Ephemeral Burst & Purge:** Inference requests pass `"keep_alive": 0` to Ollama, causing the model to unload from VRAM within milliseconds of generation completion.

---

## Environment Configuration

Configure the following environment variables in your shell (`.bashrc`, `.zshrc`, or Windows Environment Variables):

```bash
# Enable 8-bit quantized KV cache to fit 8192 context within 8 GB VRAM
export OLLAMA_KV_CACHE_TYPE=q8_0

# Enable Flash Attention in Ollama
export OLLAMA_FLASH_ATTENTION=1

# Optional: Override Antigravity brain directory location
# export ANTIGRAVITY_BRAIN_DIR="C:/Users/<User>/.gemini/antigravity/brain"
```

---

## Installation

Install in editable mode with development dependencies:

```bash
git clone https://github.com/AugustyaB/LLM-Optimizer.git
cd LLM-Optimizer
pip install -e ".[dev]"
```

Verify installation:

```bash
pcompile --help
pytest -v
```

---

## Daily CLI Usage

`pcompile` automatically discovers your most recent Antigravity conversation directory in under 15 ms, parses `.system_generated/logs/transcript.jsonl`, prunes completed phase files, extracts active code signatures, and pipes the compiled prompt directly to your system clipboard (`clip.exe`, `wl-copy`, or `xclip`).

### 1. STEP Mode (`--step` / `-st`)
Use when you want to proceed to the next phase in `task.md` or `implementation_plan.md`:
```bash
pcompile --step
```
- **Context Window:** 2048 tokens (`num_predict: 250`).
- **Output:** A compact ~120-word Phase-Transition Directive locking working set to active phase files, evicting completed files, and anchoring pinned user invariants.

### 2. REFINE Mode (`--refine` / `-r` or default)
Use when implementing features, fixing bugs, or making mid-phase adjustments:
```bash
pcompile "Implement buffered stream reading with max 64KB chunks"
```
Or explicitly:
```bash
pcompile -r "Add thread-safe double-checked locking on the cache"
```
- **Context Window:** 4096 tokens (`num_predict: 500`).
- **Output:** A structured ~300-word specification containing `[ROLE & OBJECTIVE]`, `[INTENT & ARCHITECTURE ALIGNMENT]`, `[INFERRED CODING EDGE CASES]`, `[CONTEXT SCOPE & FILE EVICTION LOCK]`, and bottom-anchored `[PINNED_USER_INVARIANTS]`.

### 3. ARCHITECT Mode (`--arch` / `-a`)
Use when designing, auditing, or overhauling system blueprints:
```bash
pcompile --arch "Design 5-phase modular implementation plan for telemetry harvester"
```
- **Context Window:** 8192 tokens (`num_predict: 900`).
- **Output:** A comprehensive ~700-word blueprint compiler prompt that forces the target AI to output phased specifications where every phase declares `Allowed_Working_Set`, `Evict_On_Complete`, `Inferred_Edge_Cases`, and `Verification_Gate`.

### 4. HANDOFF Mode (`--handoff` / `-ho`)
Use when a conversation has exceeded 35 turns or accumulated context drift, to bootstrap a clean, drift-free chat:
```bash
pcompile --handoff
```
- **Context Window:** 4096 tokens (`num_predict: 600`).
- **Output:** A clean session handoff directive summarizing completed milestones, locking active phase files with AST/Java/TS/SQL signatures, and pruning stale history.

### 5. Side-by-Side Comparison Mode (`--compare`)
Compare the deterministic CPU prompt against the trained Qwen3-8B neural output side-by-side:
```bash
pcompile --compare "Optimize database connection pooling with health check pings"
```
- Outputs both **[1] Deterministic CPU Baseline** and **[2] Trained Qwen3-8B Neural Output** in the terminal.
- Automatically copies the Neural Output to the system clipboard.

### 6. Interactive Session Picker (`--select` / `-s`)
Pick from the top 10 most recently modified Antigravity sessions:
```bash
pcompile --select
```

### 7. Dry Harvest / Instant Offline Mode (`--dry-harvest` / `--offline`)
Harvest telemetry, evict stale files, and compile prompt deterministically via CPU standard library in `< 30 ms` without waking the GPU or calling Ollama:
```bash
pcompile --dry-harvest "Refine memory allocator"
```

---

## Architectural Highlights

### Deep Hugging Face Solution Constraint Extractor (`training/hf_extractor.py`)
Inspects both instructions and reference solutions in Hugging Face coding datasets to infer task-specific constraints:
- **Exception Classes:** Detects specific failure modes (`TimeoutError`, `KeyError`, `ZeroDivisionError`).
- **Concurrency & Async:** Captures task cancellation token propagation and event loop non-blocking guards.
- **Memory & Streaming:** Infers generator chunking, buffer limits, and $O(1)$ auxiliary space complexity constraints.
- **Database Transactions:** Extracts atomic transaction wrapping and rollback guarantees.
- **Type Safety & Data Invariants:** Enforces static typing, dataclasses, and immutable representations.

### Multi-Language Code Signature Extractor (`pcompile/signature_extractor.py`)
Extracts top-level classes, methods, and functions in under 5 ms:
- **Python (`.py`):** Uses standard library `ast.parse` to extract class definitions, methods, and typed parameters.
- **Java (`.java`):** Regex-based extraction of Spring Boot annotations (`@Service`, `@RestController`), class declarations, and method signatures.
- **TypeScript (`.ts`, `.tsx`):** Extracts `interface`, `type`, `class`, and exported functions.
- **SQL (`.sql`):** Extracts `CREATE TABLE`, `CREATE INDEX`, and `CREATE VIEW` headers.

### Hybrid Sandwich Compiler (`pcompile/sandwich.py`)
Generative LLM outputs are post-processed through a deterministic CPU sandwich:
1. Strips any leading `<think>...</think>` tags from Qwen3 models.
2. Strips any hallucinated file lists or corrupted invariant blocks from the LLM output.
3. Injects the CPU-verified `[EVICT_FROM_WORKING_SET]` and `[ACTIVE_PHASE_FILES_ONLY]` locks.
4. Appends `[ACTIVE_FILE_SIGNATURES]` and anchors `[PINNED_USER_INVARIANTS]` at the bottom.

### Dual-Directory Markdown Resolution (`pcompile/evictor.py`)
Prioritizes `task.md` and `implementation_plan.md` in the current `project_dir` (repository root), gracefully falling back to `session_dir` (`brain/<conversation_id>/`).

---

## Model Verification & A/B Benchmark Suite

Run post-training evaluation across 4 verification gates:

```bash
python -m training.evaluate_model
```

### Verification Gates:
1. **Gate 1 (Loss & Artifact Check):** Verifies `outputs/Modelfile`, `.gguf` weight exports, and confirms training loss convergence (`final_loss < initial_loss`) from `outputs/training_metrics.json`.
2. **Gate 2 (Unseen Prompt Structural Audit):** Runs 4 unseen test prompts (`STEP`, `REFINE`, `ARCHITECT`, `HANDOFF` across Python, Java, TypeScript, SQL) against the model, verifying required markdown markers, zero conversational preamble, and `<think>` suppression.
3. **Gate 3 (Neural Novel Technical Token Ratio):** Compares the model's generated edge cases against the static CPU baseline, calculating the ratio of novel technical tokens to ensure genuine engineering value-add ($\ge 0.15$).
4. **Gate 4 (Post-Inference VRAM Purge Check):** Runs `check_vram()` to ensure `"keep_alive": 0` unloaded model weights back to 0 MB VRAM retention.

---

## QLoRA Fine-Tuning Pipeline

Train a 4-bit quantized adapter (`unsloth/Qwen3-8B-unsloth-bnb-4bit`) on 600 diverse software engineering training pairs:

### 1. Synthesize & Validate Training Dataset
```bash
python -m training.synthesize_dataset --output training/dataset.jsonl --use-hf
```
- Generates 600 unique training pairs (150 `STEP`, 200 `REFINE`, 150 `ARCHITECT`, 100 `HANDOFF`) across 25 polyglot software domains with deep HF solution constraint extraction and zero synthetic numbering in outputs.

### 2. Pre-Flight VRAM Check & Smoke Test
```bash
python -m training.train_qlora --smoke-test
```
- Verifies VRAM usage against the `< 350 MiB` threshold.
- Formats all 600 records into ChatML format with `/no_think` reasoning suppression.
- Validates RTX 4070 hyperparameters (`r=16`, `alpha=32`, `use_rslora=True`, `optim="adamw_8bit"`).
- Renders `outputs/Modelfile` and records pre-flight metrics.

### 3. Full Training & Automated GGUF Export
```bash
python -m training.train_qlora --output-dir outputs
```
- Trains for 350 steps inside the 8 GB VRAM budget with response-only loss (`train_on_responses_only`).
- Automatically exports merged weights to `outputs/gguf/pcompile-q4_k_m.gguf`.
- Saves telemetry loss metrics to `outputs/training_metrics.json`.
- Registers model in Ollama:
  ```bash
  ollama create pcompile -f outputs/Modelfile
  ```

---

## Repository Architecture

```text
pcompile/
├── pyproject.toml               # Package build, CLI script entry, and pytest config
├── .gitignore                   # Excludes .venv/, *.gguf, outputs/, caches
├── task.md                      # Live 7-phase milestone execution tracker
├── README.md                    # System architecture and usage guide
├── pcompile/
│   ├── __init__.py              # Package version (0.1.0)
│   ├── config.py                # Hardware constants, token budgets, and brain auto-resolver
│   ├── session_selector.py      # Sub-15ms session discovery and top-10 interactive picker
│   ├── harvester.py             # 35-turn two-zone sliding window (Zone A & B) and invariant pinner
│   ├── evictor.py               # Active working-set file classifier and dual-directory context evictor
│   ├── signature_extractor.py   # Multi-language AST/regex code signature extractor (<5ms CPU)
│   ├── modes.py                 # STEP, REFINE, ARCHITECT, and HANDOFF payload compilers
│   ├── sandwich.py              # Hybrid Sandwich compiler post-processor (<think> tag stripping)
│   └── cli.py                   # Ephemeral CLI with --compare, Ollama client (keep_alive=0), clipboard pipe
├── training/
│   ├── __init__.py
│   ├── domain_bank.py           # 25 polyglot domain definitions and coprime prompt templates
│   ├── hf_extractor.py          # Deep Hugging Face solution constraint extractor (<180 lines)
│   ├── vram_guard.py            # Pre-flight nvidia-smi VRAM check (< 350 MiB threshold)
│   ├── synthesize_dataset.py    # 600-pair hybrid HF + dynamic edge-case dataset engine
│   ├── train_qlora.py           # Unsloth 4-bit QLoRA trainer (Qwen3-8B) + Q4_K_M GGUF exporter
│   ├── evaluate_model.py        # 4-gate verification and scorecard evaluation suite
│   ├── Modelfile.template       # Ollama Modelfile configuration with /no_think suppression
│   └── dataset.jsonl            # 600 validated training pairs
└── tests/
    ├── conftest.py              # Mock Antigravity brain/ UUID trees and JSONL fixtures
    ├── test_phase1_selector.py  # Unit tests for session selection and brain path resolver
    ├── test_phase2_harvester.py # Unit tests for sliding window, eviction, and invariant pinning
    ├── test_phase3_modes.py     # Unit tests for mode specifications and dataset schema
    ├── test_phase4_training.py  # Unit tests for VRAM guard, ChatML, and smoke test
    ├── test_phase5_cli.py       # Unit tests for CLI flags, Ollama keep_alive=0, and clipboard pipe
    ├── test_phase6_perfection.py# Unit tests for AST signatures, handoff, sandwich, and anti-monolith
    └── test_phase7_evaluation.py# Unit tests for HF extractor, model evaluator, and --compare
```

---

## Verification & Testing

Execute the complete test suite across all 7 phases:

```bash
pytest -v
```
All 95 tests across 7 test suites run in ~0.7 seconds using in-memory mock Antigravity telemetry fixtures with a 100% pass rate.
