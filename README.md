# `pcompile`: Ephemeral Antigravity Prompt & Context Compiler

A zero-idle-footprint prompt & context compiler and QLoRA fine-tuning pipeline designed for local developer pairs working with Google Antigravity and Gemini 3.8 Flash.

`pcompile` reads local Google Antigravity session telemetry (`brain/<conversation_id>/`), prunes stale file context, pins recent user invariants, infers unspoken coding edge cases, and compiles raw thoughts into drift-proof prompt specifications.

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

`pcompile` automatically discovers your most recent Antigravity conversation directory in under 15 ms, parses `.system_generated/logs/transcript.jsonl`, prunes completed phase files, and pipes the compiled prompt directly to your system clipboard (`clip.exe`, `wl-copy`, or `xclip`).

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

### 4. Interactive Session Picker (`--select` / `-s`)
Pick from the top 10 most recently modified Antigravity sessions:
```bash
pcompile --select
```

### 5. Dry Harvest / Instant Offline Mode (`--dry-harvest` / `--offline`)
Harvest telemetry, evict stale files, and compile prompt deterministically via CPU standard library in `< 30 ms` without waking the GPU or calling Ollama:
```bash
pcompile --dry-harvest "Refine memory allocator"
```

---

## QLoRA Fine-Tuning Pipeline

Train a 4-bit quantized adapter (`unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit`) on 500 diverse software engineering training pairs:

### 1. Synthesize & Validate Training Dataset
```bash
python -m training.synthesize_dataset --output training/dataset.jsonl
```
- Generates 500 unique training pairs (150 `STEP`, 200 `REFINE`, 150 `ARCHITECT`) across 30 software domains with dynamic edge-case inference in under 10 ms.

### 2. Pre-Flight VRAM Check & Smoke Test
```bash
python -m training.train_qlora --smoke-test
```
- Verifies VRAM usage against the `< 350 MiB` threshold.
- Formats all 500 records into ChatML format.
- Validates RTX 4070 hyperparameters (`r=16`, `alpha=32`, `use_rslora=True`, `optim="adamw_8bit"`).
- Renders `outputs/Modelfile`.

### 3. Full Training & Automated GGUF Export
```bash
python -m training.train_qlora --output-dir outputs
```
- Trains for 350 steps inside the 8 GB VRAM budget.
- Automatically exports merged weights to `outputs/gguf/pcompile-q4_k_m.gguf`.
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
├── task.md                      # Live 5-phase milestone execution tracker
├── README.md                    # System architecture and usage guide
├── pcompile/
│   ├── __init__.py              # Package version (0.1.0)
│   ├── config.py                # Hardware constants, token budgets, and brain auto-resolver
│   ├── session_selector.py      # Sub-15ms session discovery and top-10 interactive picker
│   ├── harvester.py             # 35-turn two-zone sliding window (Zone A & B) and invariant pinner
│   ├── evictor.py               # Active working-set file classifier and context evictor
│   ├── modes.py                 # STEP, REFINE, ARCHITECT payload compilers
│   └── cli.py                   # Ephemeral Ollama client (keep_alive=0) and clipboard pipe
├── training/
│   ├── __init__.py
│   ├── vram_guard.py            # Pre-flight nvidia-smi VRAM check (< 350 MiB threshold)
│   ├── synthesize_dataset.py    # 500-pair hybrid HF + dynamic edge-case dataset engine
│   ├── train_qlora.py           # Unsloth 4-bit QLoRA trainer + Q4_K_M GGUF exporter
│   ├── Modelfile.template       # Ollama Modelfile configuration
│   └── dataset.jsonl            # 500 validated training pairs
└── tests/
    ├── conftest.py              # Mock Antigravity brain/ UUID trees and JSONL fixtures
    ├── test_phase1_selector.py  # Unit tests for session selection and brain path resolver
    ├── test_phase2_harvester.py # Unit tests for sliding window, eviction, and invariant pinning
    ├── test_phase3_modes.py     # Unit tests for mode specifications and dataset schema
    ├── test_phase4_training.py  # Unit tests for VRAM guard, ChatML, and smoke test
    └── test_phase5_cli.py       # Unit tests for CLI flags, Ollama keep_alive=0, and clipboard pipe
```

---

## Verification & Testing

Execute the complete test suite across all 5 phases:

```bash
pytest -v
```
All test suites run in under 0.5 seconds using in-memory mock Antigravity telemetry fixtures.
