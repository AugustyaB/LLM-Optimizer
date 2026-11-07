"""Hardware-Isolated Unsloth 4-Bit QLoRA Training and Q4_K_M GGUF Exporter.

Why: Fine-tunes Qwen3-8B with ChatML response-only loss on 8GB VRAM with zero idle footprint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training.synthesize_dataset import generate_dataset, validate_dataset
from training.vram_guard import check_vram

DEFAULT_BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
FALLBACK_BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"


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
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def format_training_record(record: dict[str, Any]) -> str:
    """Format dataset record into ChatML format with /no_think reasoning suppression."""
    instruction = record.get("instruction", "").strip()
    inp, out = record.get("input", "").strip(), record.get("output", "").strip()
    return (
        "<|im_start|>system\n"
        "You are pcompile, an ephemeral prompt and context compiler for Google Antigravity and Gemini 3.8 Flash. "
        "Output ONLY compiled prompt specifications in STEP, REFINE, ARCHITECT, or HANDOFF mode. "
        "Never include conversational filler, meta-chatter, apologies, or preambles.<|im_end|>\n"
        f"<|im_start|>user\n[INSTRUCTION]\n{instruction}\n\n[INPUT]\n{inp} /no_think<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>{out}<|im_end|>"
    )


def render_modelfile(template_path: Path, output_path: Path, gguf_path: Path) -> Path:
    """Render Ollama Modelfile from template with configured GGUF model path."""
    if not template_path.is_file():
        raise FileNotFoundError(f"Modelfile template not found: {template_path}")
    rendered = template_path.read_text(encoding="utf-8").replace("{{ GGUF_PATH }}", gguf_path.as_posix())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def load_or_generate_dataset(dataset_path: Path, use_hf: bool = False) -> list[dict[str, Any]]:
    """Ensure dataset.jsonl exists; auto-generate and validate if missing."""
    if not dataset_path.is_file():
        records = generate_dataset(output_path=dataset_path, use_hf=use_hf)
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
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
    vram_status = check_vram() if not skip_vram_check else None
    if vram_status:
        print(vram_status.format_diagnostic())

    records = load_or_generate_dataset(dataset_path, use_hf=use_hf)
    for idx, r in enumerate(records):
        fmt = format_training_record(r)
        assert "<|im_start|>system" in fmt and "/no_think" in fmt and "<think>" in fmt, f"Record {idx} invalid ChatML"

    tpl = template_path or Path("training/Modelfile.template")
    modelfile_out = output_dir / "Modelfile"
    render_modelfile(tpl, modelfile_out, output_dir / "gguf" / "pcompile-q4_k_m.gguf")

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
    """Run full Unsloth QLoRA fine-tuning with response-only loss and export to Q4_K_M GGUF."""
    if not skip_vram_check:
        check_vram(enforce=True)

    records = load_or_generate_dataset(dataset_path, use_hf=use_hf)
    try:
        from datasets import Dataset  # type: ignore
        from transformers import TrainingArguments  # type: ignore
        from trl import SFTTrainer  # type: ignore
        from unsloth import FastLanguageModel  # type: ignore
        from unsloth.chat_templates import train_on_responses_only  # type: ignore
    except ImportError as err:
        raise RuntimeError(f"Unsloth dependencies missing: {err}. Run with --smoke-test.") from err

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model, max_seq_length=config.max_seq_length, load_in_4bit=config.load_in_4bit
        )
    except Exception:
        print(f"Base model {config.base_model} unavailable; falling back to {FALLBACK_BASE_MODEL}...")
        config.base_model = FALLBACK_BASE_MODEL
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.base_model, max_seq_length=config.max_seq_length, load_in_4bit=config.load_in_4bit
        )

    model = FastLanguageModel.get_peft_model(
        model, r=config.r, target_modules=config.target_modules, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, bias=config.bias,
        use_gradient_checkpointing=config.use_gradient_checkpointing, use_rslora=config.use_rslora,
    )

    train_dataset = Dataset.from_dict({"text": [format_training_record(r) for r in records]})
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"), per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps, learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type, optim=config.optim, bf16=config.bf16, max_steps=config.max_steps,
        warmup_steps=config.warmup_steps, weight_decay=config.weight_decay, logging_steps=10, save_strategy="steps",
        save_steps=100, report_to="none",
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_dataset, dataset_text_field="text",
        max_seq_length=config.max_seq_length, args=training_args,
    )
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train()

    gguf_output_dir = output_dir / "gguf"
    model.save_pretrained_gguf(str(gguf_output_dir), tokenizer, quantization_method="q4_k_m")

    # Resolve exported gguf file
    exported = next(gguf_output_dir.glob("*.gguf"), gguf_output_dir / "pcompile-q4_k_m.gguf") if gguf_output_dir.is_dir() else (gguf_output_dir / "pcompile-q4_k_m.gguf")
    render_modelfile(Path("training/Modelfile.template"), output_dir / "Modelfile", exported)


def main() -> None:
    """CLI entry point for QLoRA training pipeline."""
    parser = argparse.ArgumentParser(description="Unsloth 4-bit QLoRA fine-tuning and GGUF export pipeline.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model identifier.")
    parser.add_argument("--dataset", default="training/dataset.jsonl", help="Dataset path.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--max-steps", type=int, default=350, help="Maximum training steps.")
    parser.add_argument("--smoke-test", action="store_true", help="Run pre-flight verification without training.")
    parser.add_argument("--skip-vram-check", action="store_true", help="Bypass VRAM threshold check.")
    parser.add_argument("--use-hf", action="store_true", help="Fetch prompts from Hugging Face if dataset missing.")
    args = parser.parse_args()

    cfg = QLoRAConfig(base_model=args.base_model, max_steps=args.max_steps)
    if args.smoke_test:
        run_smoke_test(cfg, Path(args.dataset), Path(args.output_dir), args.skip_vram_check, use_hf=args.use_hf)
    else:
        train_and_export(cfg, Path(args.dataset), Path(args.output_dir), args.skip_vram_check, use_hf=args.use_hf)


if __name__ == "__main__":
    main()
