"""Thorough unit tests for Phase 4: Hardware-Isolated VRAM Guard, ChatML, and QLoRA Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from training.train_qlora import (
    QLoRAConfig,
    format_training_record,
    load_or_generate_dataset,
    render_modelfile,
    run_smoke_test,
)
from training.vram_guard import (
    GPUProcessInfo,
    VRAMStatus,
    check_vram,
    query_gpu_processes,
    query_nvidia_vram,
)


class TestVRAMGuard:
    """Test NVIDIA VRAM guard threshold detection, process parsing, and diagnostics."""

    def test_vram_guard_safe_threshold(self) -> None:
        """Used VRAM below 350 MiB must be marked safe."""

        def mock_runner(cmd: list[str]) -> str:
            if "query-gpu" in " ".join(cmd):
                return "150, 8188\n"
            return ""

        status = check_vram(threshold_mb=350, run_cmd=mock_runner)
        assert status.is_safe is True
        assert status.used_mb == 150
        assert status.total_mb == 8188
        assert "VRAM GUARD OK" in status.format_diagnostic()

    def test_vram_guard_exceeds_threshold_diagnostic(self) -> None:
        """Used VRAM above 350 MiB must trigger alert with process list and remediation steps."""

        def mock_runner(cmd: list[str]) -> str:
            cmd_str = " ".join(cmd)
            if "query-gpu" in cmd_str:
                return "1450, 8188\n"
            if "query-compute-apps" in cmd_str:
                return "1024, chrome.exe, 650\n2048, opera.exe, 400\n"
            return ""

        status = check_vram(threshold_mb=350, run_cmd=mock_runner)
        assert status.is_safe is False
        assert status.used_mb == 1450

        diag = status.format_diagnostic()
        assert "VRAM GUARD ALERT" in diag
        assert "1450 MiB exceeds threshold of 350 MiB" in diag
        assert "AMD Radeon 780M" in diag
        assert "chrome.exe" in diag
        assert "opera.exe" in diag

    def test_vram_guard_enforce_raises_runtime_error(self) -> None:
        """When enforce=True and VRAM exceeds threshold, RuntimeError must be raised."""

        def mock_runner(cmd: list[str]) -> str:
            return "800, 8188\n"

        with pytest.raises(RuntimeError, match="VRAM GUARD ALERT"):
            check_vram(threshold_mb=350, enforce=True, run_cmd=mock_runner)

    def test_vram_guard_missing_or_failed_nvidia_smi(self) -> None:
        """When nvidia-smi fails, guard gracefully falls back to simulated safe mode."""

        def failing_runner(cmd: list[str]) -> str:
            raise FileNotFoundError("nvidia-smi not found")

        status = check_vram(threshold_mb=350, run_cmd=failing_runner)
        assert status.is_safe is True
        assert "nvidia-smi query unavailable" in status.diagnostic_message


class TestChatMLAndModelfile:
    """Test ChatML prompt formatting and Ollama Modelfile rendering."""

    def test_format_training_record_chatml(self) -> None:
        """Record must format conforming to ChatML standard with system, user, and assistant blocks."""
        record = {
            "instruction": "Compile a prompt in REFINE mode.",
            "input": "User request: optimize parser.",
            "output": "### [ROLE & OBJECTIVE]\nOptimize parser safely.\n\n[PINNED_USER_INVARIANTS]\n- Rule 1",
        }
        formatted = format_training_record(record)

        assert "<|im_start|>system" in formatted
        assert "You are pcompile, an ephemeral prompt and context compiler" in formatted
        assert "<|im_start|>user" in formatted
        assert "[INSTRUCTION]\nCompile a prompt in REFINE mode." in formatted
        assert "[INPUT]\nUser request: optimize parser." in formatted
        assert "<|im_start|>assistant" in formatted
        assert "### [ROLE & OBJECTIVE]\nOptimize parser safely." in formatted
        assert formatted.endswith("<|im_end|>")

    def test_render_modelfile(self, tmp_path: Path) -> None:
        """Template must render with configured GGUF path and sampling parameters."""
        tpl_path = tmp_path / "Modelfile.template"
        tpl_path.write_text(
            "FROM {{ GGUF_PATH }}\nPARAMETER temperature 0.2\nPARAMETER num_ctx 4096\nSYSTEM test\n",
            encoding="utf-8",
        )

        out_path = tmp_path / "outputs" / "Modelfile"
        gguf_target = tmp_path / "outputs" / "gguf" / "pcompile-q4_k_m.gguf"

        res = render_modelfile(tpl_path, out_path, gguf_target)
        assert res == out_path
        assert out_path.is_file()

        content = out_path.read_text(encoding="utf-8")
        assert f"FROM {gguf_target.as_posix()}" in content
        assert "PARAMETER temperature 0.2" in content
        assert "PARAMETER num_ctx 4096" in content

    def test_render_modelfile_missing_template_raises(self, tmp_path: Path) -> None:
        """Missing Modelfile.template must raise FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.template"
        with pytest.raises(FileNotFoundError, match="Modelfile template not found"):
            render_modelfile(nonexistent, tmp_path / "out", tmp_path / "gguf")


class TestQLoRAPipelineSmokeTest:
    """Test QLoRA hyperparameter configuration and end-to-end smoke test mode."""

    def test_qlora_config_hardware_hyperparameters(self) -> None:
        """Verify RTX 4070 Laptop (8GB VRAM) and Ryzen 9 7940HS hyperparameters."""
        cfg = QLoRAConfig()

        assert cfg.base_model in ("unsloth/Qwen3-8B-unsloth-bnb-4bit", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
        assert cfg.max_seq_length == 4096
        assert cfg.load_in_4bit is True
        assert cfg.r == 16
        assert cfg.lora_alpha == 32
        assert cfg.per_device_train_batch_size == 1
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.optim == "adamw_8bit"
        assert cfg.bf16 is True
        assert cfg.max_steps == 350
        assert cfg.learning_rate == 2e-4
        assert len(cfg.target_modules) == 7

    def test_run_smoke_test_full_verification(self, tmp_path: Path) -> None:
        """Smoke test must validate dataset, ChatML formatting, and Modelfile without GPU training."""
        config = QLoRAConfig()
        dataset_path = tmp_path / "dataset.jsonl"
        output_dir = tmp_path / "outputs"

        # Provide template
        tpl_path = tmp_path / "Modelfile.template"
        tpl_path.write_text("FROM {{ GGUF_PATH }}\nPARAMETER temperature 0.2\n", encoding="utf-8")

        result = run_smoke_test(
            config=config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            skip_vram_check=True,
            template_path=tpl_path,
        )

        assert result["status"] == "SMOKE_TEST_PASSED"
        assert result["records_validated"] in (500, 600)
        assert (output_dir / "Modelfile").is_file()
        assert result["hyperparameters"]["max_steps"] == 350
