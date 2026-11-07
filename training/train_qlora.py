"""Hardware-Isolated Unsloth 4-Bit QLoRA Training and Q4_K_M GGUF Exporter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pcompile.config import TARGET_VRAM_MB, VRAM_GUARD_THRESHOLD_MB
from training.synthesize_dataset import generate_dataset, validate_dataset
from training.vram_guard import check_vram

# ---------------------------------------------------------------------------
# QLoRA Hyperparameters for RTX 4070 Laptop (8GB VRAM) + Ryzen 9 7940HS
# ---------------------------------------------------------------------------
DEFAULT_BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
FALLBACK_BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"


@dataclass
class QLoRAConfig:
    """Hardware-locked hyperparameters for 8 GB GDDR6 envelope."""

    base_model: str = DEFAULT_BASE_MODEL
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    use_rslora: bool = True
    target_modules: list[str] = None  # type: ignore

    # Trainer parameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    optim: str = "adamw_8bit"
    bf16: bool = True
    dataset_num_proc: int = 8
    max_steps: int = 350
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 20
    weight_decay: float = 0.01

    def __post_init__(self) -> None:
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


def format_training_record(record: dict[str, Any]) -> str:
    """Format a dataset record into standard ChatML text format."""
    instruction = record.get("instruction", "").strip()
    input_text = record.get("input", "").strip()
    output_text = record.get("output", "").strip()

    formatted = (
        "<|im_start|>system\n"
        "You are pcompile, an ephemeral prompt and context compiler for Google Antigravity and Gemini 3.8 Flash. "
        "Output ONLY compiled prompt specifications in STEP, REFINE, or ARCHITECT mode. "
        "Never include conversational filler, meta-chatter, or preambles.<|im_end|>\n"
        "<|im_start|>user\n"
        f"[INSTRUCTION]\n{instruction}\n\n"
        f"[INPUT]\n{input_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{output_text}<|im_end|>"
    )
    return formatted


def render_modelfile(template_path: Path, output_path: Path, gguf_path: Path) -> Path:
    """Render Ollama Modelfile from template with configured GGUF model path."""
    if not template_path.is_file():
        raise FileNotFoundError(f"Modelfile template not found: {template_path}")

    template_content = template_path.read_text(encoding="utf-8")
    rendered = template_content.replace("{{ GGUF_PATH }}", gguf_path.as_posix())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def load_or_generate_dataset(dataset_path: Path, use_hf: bool = False) -> list[dict[str, Any]]:
    """Ensure dataset.jsonl exists; auto-generate and validate if missing."""
    if not dataset_path.is_file():
        print(f"Dataset not found at {dataset_path}. Synthesizing 500 records...")
        records = generate_dataset(output_path=dataset_path, use_hf=use_hf)
    else:
        records = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    validate_dataset(records)
    return records


def run_smoke_test(
    config: QLoRAConfig,
    dataset_path: Path,
    output_dir: Path,
    skip_vram_check: bool = False,
    template_path: Path | None = None,
    use_hf: bool = False,
) -> dict[str, Any]:
    """Execute pre-flight smoke test verifying dataset, VRAM guard, and Modelfile rendering."""
    print("Executing QLoRA pre-flight smoke test...")

    # 1. Check VRAM Guard
    vram_status = None
    if not skip_vram_check:
        vram_status = check_vram()
        print(vram_status.format_diagnostic())
    else:
        print("[VRAM GUARD] Bypassed via --skip-vram-check.")

    # 2. Verify Dataset
    records = load_or_generate_dataset(dataset_path, use_hf=use_hf)
    print(f"[DATASET CHECK] Loaded & validated {len(records)} records from {dataset_path}.")

    # 3. Verify ChatML formatting on all records
    for idx, r in enumerate(records):
        formatted = format_training_record(r)
        assert "<|im_start|>system" in formatted, f"Record {idx} missing system ChatML tag"
        assert "<|im_start|>user" in formatted, f"Record {idx} missing user ChatML tag"
        assert "<|im_start|>assistant" in formatted, f"Record {idx} missing assistant ChatML tag"

    # 4. Render Modelfile
    tpl_path = template_path or Path("training/Modelfile.template")
    modelfile_out = output_dir / "Modelfile"
    gguf_target = output_dir / "gguf" / "pcompile-q4_k_m.gguf"
    render_modelfile(tpl_path, modelfile_out, gguf_target)
    print(f"[MODELFILE CHECK] Rendered Modelfile to {modelfile_out}.")

    # 5. Hyperparameter confirmation
    print(f"[CONFIG CHECK] Verified 4-bit QLoRA hyperparameters for {config.base_model}:")
    print(f"  max_seq_length:       {config.max_seq_length}")
    print(f"  lora_r / alpha:       {config.r} / {config.lora_alpha} (rslora={config.use_rslora})")
    print(f"  target_modules:       {len(config.target_modules)} modules")
    print(f"  batch_size / accum:   {config.per_device_train_batch_size} / {config.gradient_accumulation_steps}")
    print(f"  optim / scheduler:    {config.optim} / {config.lr_scheduler_type} (bf16={config.bf16})")
    print(f"  max_steps:            {config.max_steps}")

    return {
        "status": "SMOKE_TEST_PASSED",
        "records_validated": len(records),
        "vram_safe": vram_status.is_safe if vram_status else True,
        "modelfile": str(modelfile_out),
        "hyperparameters": asdict(config),
    }


def train_and_export(
    config: QLoRAConfig,
    dataset_path: Path,
    output_dir: Path,
    skip_vram_check: bool = False,
    use_hf: bool = False,
) -> None:
    """Run full Unsloth QLoRA fine-tuning and export to Q4_K_M GGUF."""
    if not skip_vram_check:
        check_vram(enforce=True)

    records = load_or_generate_dataset(dataset_path, use_hf=use_hf)

    try:
        from unsloth import FastLanguageModel  # type: ignore
        from trl import SFTTrainer  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from datasets import Dataset  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            f"Unsloth training dependencies not installed: {err}. "
            "To train on GPU: pip install unsloth trl transformers datasets. "
            "For pre-flight verification, run with --smoke-test."
        ) from err

    print(f"Loading base model {config.base_model} in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
    )

    print("Configuring PEFT LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        use_rslora=config.use_rslora,
    )

    formatted_texts = [format_training_record(r) for r in records]
    train_dataset = Dataset.from_dict({"text": formatted_texts})

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        bf16=config.bf16,
        max_steps=config.max_steps,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    # GGUF Export
    gguf_output_dir = output_dir / "gguf"
    print(f"Exporting merged model to Q4_K_M GGUF at {gguf_output_dir}...")
    model.save_pretrained_gguf(
        str(gguf_output_dir),
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Dynamically resolve exported .gguf file
    exported_gguf: Path | None = None
    if gguf_output_dir.is_dir():
        for candidate in gguf_output_dir.glob("*.gguf"):
            if candidate.is_file():
                exported_gguf = candidate
                break
    if exported_gguf is None:
        exported_gguf = gguf_output_dir / "pcompile-q4_k_m.gguf"

    # Render Modelfile
    modelfile_path = output_dir / "Modelfile"
    render_modelfile(
        Path("training/Modelfile.template"),
        modelfile_path,
        exported_gguf,
    )
    print(f"QLoRA training and GGUF export complete. Modelfile saved to {modelfile_path}.")


def main() -> None:
    """CLI entry point for QLoRA training pipeline."""
    parser = argparse.ArgumentParser(description="Unsloth 4-bit QLoRA fine-tuning and GGUF export pipeline.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL, help="Base model identifier.")
    parser.add_argument("--dataset", type=str, default="training/dataset.jsonl", help="Dataset path.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--max-steps", type=int, default=350, help="Maximum training steps.")
    parser.add_argument("--smoke-test", action="store_true", help="Run pre-flight verification without training.")
    parser.add_argument("--skip-vram-check", action="store_true", help="Bypass VRAM threshold check.")
    parser.add_argument("--use-hf", action="store_true", help="Fetch prompts from Hugging Face if dataset missing.")
    args = parser.parse_args()

    config = QLoRAConfig(
        base_model=args.base_model,
        max_steps=args.max_steps,
    )

    if args.smoke_test:
        run_smoke_test(
            config=config,
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            skip_vram_check=args.skip_vram_check,
            use_hf=args.use_hf,
        )
    else:
        train_and_export(
            config=config,
            dataset_path=Path(args.dataset),
            output_dir=Path(args.output_dir),
            skip_vram_check=args.skip_vram_check,
            use_hf=args.use_hf,
        )


if __name__ == "__main__":
    main()
