"""Pinned offline Qwen3 runtime for the compact tagged protocol."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from transformers import (  # type: ignore[import-untyped]
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from foundry.synthesis.realization.compact_contracts import COMPACT_SURFACE_SYSTEM_PROMPT
from foundry.synthesis.realization.compact_prompting import serialize_compact_request
from foundry.synthesis.realization.compact_request import PreparedCompactRequest
from foundry.synthesis.realization.compact_smoke_contract import CompactSmokeConfig
from foundry.synthesis.realization.local_runtime import (
    GeneratedBeam,
    RuntimeGeneration,
    RuntimeMetadata,
)


class _ClosingTagCriteria(StoppingCriteria):  # type: ignore[misc]
    """Finish each beam immediately after its complete terminal closing Q tag."""

    def __init__(self, token_ids: tuple[int, ...]) -> None:
        if not token_ids:
            raise ValueError("closing tag token sequence cannot be empty")
        self.token_ids = token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: object
    ) -> torch.BoolTensor:
        del scores
        width = len(self.token_ids)
        target = torch.tensor(self.token_ids, dtype=input_ids.dtype, device=input_ids.device)
        if input_ids.shape[1] < width:
            return cast(
                torch.BoolTensor,
                torch.zeros((input_ids.shape[0],), dtype=torch.bool, device=input_ids.device),
            )
        matches = torch.all(input_ids[:, -width:] == target, dim=1)
        return cast(torch.BoolTensor, matches)


def _snapshot_path(repository_root: Path, config: CompactSmokeConfig) -> Path:
    repository_name = config.model.repo_id.replace("/", "--")
    return (
        repository_root
        / config.model.cache_root
        / "hub"
        / f"models--{repository_name}"
        / "snapshots"
        / config.model.revision
    )


class PinnedCompactQwenRealizer:
    """Offline FP16 Qwen3 with fixed three-beam compact generation."""

    def __init__(self, *, repository_root: Path, config: CompactSmokeConfig) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError("compact Qwen3 realization requires the approved CUDA desktop")
        torch.manual_seed(config.generation.seed)
        torch.cuda.manual_seed_all(config.generation.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        snapshot = _snapshot_path(repository_root, config)
        if not snapshot.is_dir() or snapshot.name != config.model.revision:
            raise RuntimeError("the exact pinned Qwen3 snapshot is unavailable")
        tokenizer_start = time.perf_counter()
        tokenizer: Any = AutoTokenizer.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
        )
        tokenizer_seconds = time.perf_counter() - tokenizer_start
        chat_template = tokenizer.chat_template
        if not isinstance(chat_template, str):
            raise RuntimeError("the pinned tokenizer has no chat template")
        chat_hash = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if chat_hash != config.chat_template_sha256:
            raise RuntimeError("Qwen3 chat-template hash differs from the frozen compact config")
        closing_ids = tokenizer.encode(
            config.generation.stopping_tag,
            add_special_tokens=False,
        )
        if not isinstance(closing_ids, list) or not all(
            isinstance(item, int) for item in closing_ids
        ):
            raise RuntimeError("Qwen3 tokenizer did not encode the compact stopping tag")
        self.stopping_criteria = StoppingCriteriaList(
            [_ClosingTagCriteria(tuple(cast(list[int], closing_ids)))]
        )
        torch.cuda.reset_peak_memory_stats()
        model_start = time.perf_counter()
        model: Any = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        model.to("cuda")
        model.eval()
        model_seconds = time.perf_counter() - model_start
        self.repository_root = repository_root
        self.config = config
        self.snapshot = snapshot
        self.tokenizer = tokenizer
        self.model = model
        self.metadata = RuntimeMetadata(
            snapshot_path=str(snapshot),
            resolved_revision=snapshot.name,
            tokenizer_load_seconds=tokenizer_seconds,
            model_load_seconds=model_seconds,
            chat_template_sha256=chat_hash,
            model_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            dtype=str(next(model.parameters()).dtype),
            device=str(next(model.parameters()).device),
        )

    def generate(self, prepared: PreparedCompactRequest) -> RuntimeGeneration:
        """Return exactly three beams; never retry or repair malformed output."""

        messages = [
            {"role": "system", "content": COMPACT_SURFACE_SYSTEM_PROMPT},
            {"role": "user", "content": serialize_compact_request(prepared.request)},
        ]
        input_ids: torch.Tensor = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        ).to("cuda")
        input_count = int(input_ids.shape[-1])
        generation = self.config.generation
        torch.manual_seed(generation.seed)
        torch.cuda.manual_seed_all(generation.seed)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs: Any = self.model.generate(
                input_ids,
                do_sample=generation.do_sample,
                num_beams=generation.num_beams,
                num_return_sequences=generation.num_return_sequences,
                max_new_tokens=generation.max_new_tokens,
                max_time=float(generation.timeout_seconds_per_ir),
                stopping_criteria=self.stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        elapsed = time.perf_counter() - start
        sequences = cast(torch.Tensor, outputs.sequences)
        if sequences.shape[0] != generation.num_return_sequences:
            raise RuntimeError("Qwen3 returned an unexpected compact beam count")
        beams: list[GeneratedBeam] = []
        for index, sequence in enumerate(sequences, start=1):
            generated_ids = sequence[input_count:]
            raw = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            nonpadding = int((generated_ids != self.tokenizer.pad_token_id).sum().item())
            beams.append(
                GeneratedBeam(
                    beam_index=index,
                    raw_text=raw,
                    raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    generated_tokens=nonpadding,
                )
            )
        return RuntimeGeneration(
            beams=tuple(beams),
            input_tokens=input_count,
            elapsed_seconds=elapsed,
            timeout_exceeded=elapsed > generation.timeout_seconds_per_ir,
        )

    def generation_config_sha256(self) -> str:
        """Hash the compact operational settings and deterministic controls."""

        payload = {
            "generation": asdict(self.config.generation),
            "model_revision": self.config.model.revision,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


__all__ = ["PinnedCompactQwenRealizer"]
