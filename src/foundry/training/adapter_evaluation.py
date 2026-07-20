"""Evaluate one PEFT adapter through the unchanged frozen development runner."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from foundry.config import EvaluationConfig, GenerationConfig, load_config
from foundry.evaluation.backends import BackendError, GenerationResult, MetricValue
from foundry.evaluation.benchmark import load_huggingface_examples
from foundry.evaluation.calibration import load_development_subset
from foundry.evaluation.manifests import load_manifest
from foundry.evaluation.prompting import ChatMessage
from foundry.evaluation.runner import run_evaluation
from foundry.evaluation.validation import (
    FINAL_MAIN_DEVELOPMENT_BASELINE,
    as_frozen_baseline_manifest,
    assert_final_evaluator_config,
    load_answer_validation_manifest,
    load_final_evaluator_manifest,
)
from foundry.training.qlora import directory_sha256


class PeftCudaBackend:
    """Pinned FP16 base plus one ignored PEFT adapter, loaded fully on CUDA."""

    def __init__(
        self,
        *,
        config: EvaluationConfig,
        model_path: Path,
        adapter_path: Path,
        expected_adapter_sha256: str,
    ) -> None:
        if config.model.device != "cuda" or config.model.dtype != "float16":
            raise BackendError("frozen adapter evaluation requires CUDA float16")
        actual_adapter_sha256 = directory_sha256(adapter_path)
        if actual_adapter_sha256 != expected_adapter_sha256:
            raise BackendError("adapter directory hash differs from frozen training evidence")
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
            peft: Any = importlib.import_module("peft")
        except ImportError as error:
            raise BackendError(
                "adapter evaluation requires the frozen training environment"
            ) from error
        if not torch.cuda.is_available():
            raise BackendError("CUDA is unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        random.seed(20260720)
        torch.manual_seed(20260720)
        torch.cuda.manual_seed_all(20260720)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.to("cuda:0")
        model = peft.PeftModel.from_pretrained(
            model, str(adapter_path), local_files_only=True, is_trainable=False
        )
        offloaded = [
            name for name, parameter in model.named_parameters() if parameter.device.type != "cuda"
        ]
        if offloaded:
            raise BackendError(f"CPU or disk offloading is prohibited: {offloaded[:3]}")
        model.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._adapter_sha256 = actual_adapter_sha256
        self._load_seconds = time.perf_counter() - load_started
        self._name = (
            f"peft-cuda:{config.model.repo_id}@{config.model.revision}"
            f"+adapter@{actual_adapter_sha256}"
        )

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        stable_id: str,
        messages: tuple[ChatMessage, ChatMessage],
        generation: GenerationConfig,
    ) -> GenerationResult:
        del stable_id
        input_ids: Any = self._tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda:0")
        attention_mask = self._torch.ones_like(input_ids)
        with self._torch.inference_mode():
            outputs: Any = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=generation.do_sample,
                max_new_tokens=generation.max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        prompt_tokens = int(input_ids.shape[-1])
        generated_ids: Any = outputs[0, prompt_tokens:]
        return GenerationResult(
            text=cast(str, self._tokenizer.decode(generated_ids, skip_special_tokens=True)),
            input_tokens=prompt_tokens,
            output_tokens=int(generated_ids.shape[-1]),
        )

    def metrics(self) -> dict[str, MetricValue]:
        return {
            "backend_load_seconds": self._load_seconds,
            "gpu_name": str(self._torch.cuda.get_device_name(0)),
            "peak_vram_allocated_bytes": int(self._torch.cuda.max_memory_allocated(0)),
            "peak_vram_reserved_bytes": int(self._torch.cuda.max_memory_reserved(0)),
            "adapter_sha256": self._adapter_sha256,
        }

    def close(self) -> None:
        """Release model memory before another arm loads."""

        del self._model
        gc.collect()
        self._torch.cuda.empty_cache()


def _progress(completed: int, total: int) -> None:
    if completed % 25 == 0 or completed == total:
        print(
            json.dumps(
                {"adapter_evaluation_completed": completed, "adapter_evaluation_total": total},
                sort_keys=True,
            ),
            flush=True,
        )


def run_adapter_evaluation(
    *,
    base_config_path: Path,
    config_path: Path,
    development_manifest_path: Path,
    source_pool_manifest_path: Path,
    source_baseline_manifest_path: Path,
    baseline_manifest_path: Path,
    model_path: Path,
    adapter_path: Path,
    expected_adapter_sha256: str,
    output_dir: Path,
    tracked_summary_path: Path,
) -> dict[str, object]:
    """Run exactly the frozen 814-example evaluation with a PEFT backend."""

    base_config = load_config(base_config_path)
    config = load_config(config_path)
    assert_final_evaluator_config(base_config, config)
    development = load_manifest(development_manifest_path, base_config)
    source_pool = load_development_subset(source_pool_manifest_path, development)
    source_baseline = load_answer_validation_manifest(
        source_baseline_manifest_path, source_pool, development
    )
    baseline = load_final_evaluator_manifest(baseline_manifest_path, source_baseline, development)
    if baseline.purpose != FINAL_MAIN_DEVELOPMENT_BASELINE or len(baseline.entries) != 814:
        raise ValueError("adapter evaluation requires the frozen 814 development IDs")
    manifest = as_frozen_baseline_manifest(baseline, source_baseline, development, config)
    backend = PeftCudaBackend(
        config=config,
        model_path=model_path,
        adapter_path=adapter_path,
        expected_adapter_sha256=expected_adapter_sha256,
    )
    try:
        examples = load_huggingface_examples(config, manifest)
        summary = run_evaluation(
            config=config,
            manifest=manifest,
            examples=examples,
            backend=backend,
            output_dir=output_dir,
            progress_callback=_progress,
        )
    finally:
        backend.close()
    result = asdict(summary)
    result["adapter_sha256"] = expected_adapter_sha256
    result["frozen_base_correct"] = 521
    result["frozen_base_accuracy"] = 521 / 814
    result["correct_delta_vs_base"] = int(result["correct_examples"]) - 521
    result["accuracy_delta_vs_base"] = float(result["accuracy"]) - 521 / 814
    result["sealed_final_accessed"] = False
    tracked_summary_path.parent.mkdir(parents=True, exist_ok=True)
    tracked_summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cast(dict[str, object], result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--development-manifest", required=True, type=Path)
    parser.add_argument("--source-pool-manifest", required=True, type=Path)
    parser.add_argument("--source-baseline-manifest", required=True, type=Path)
    parser.add_argument("--baseline-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tracked-summary", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run_adapter_evaluation(
        base_config_path=args.base_config,
        config_path=args.config,
        development_manifest_path=args.development_manifest,
        source_pool_manifest_path=args.source_pool_manifest,
        source_baseline_manifest_path=args.source_baseline_manifest,
        baseline_manifest_path=args.baseline_manifest,
        model_path=args.model_path,
        adapter_path=args.adapter,
        expected_adapter_sha256=args.adapter_sha256,
        output_dir=args.output_dir,
        tracked_summary_path=args.tracked_summary,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
