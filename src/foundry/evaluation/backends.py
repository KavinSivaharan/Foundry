"""Model backends with a deterministic fake and a pinned Hugging Face implementation."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Protocol, cast

from foundry.config import EvaluationConfig, GenerationConfig
from foundry.evaluation.prompting import ChatMessage

MetricValue = int | float | str | None


class BackendError(RuntimeError):
    """Raised when a model backend cannot satisfy the evaluation protocol."""


@dataclass(frozen=True)
class GenerationResult:
    """Generated text plus token accounting when a tokenizer is available."""

    text: str
    input_tokens: int | None
    output_tokens: int | None


class ModelBackend(Protocol):
    """Minimal backend interface consumed by the evaluation runner."""

    @property
    def name(self) -> str:
        """Return a stable backend name."""

    def generate(
        self,
        stable_id: str,
        messages: tuple[ChatMessage, ChatMessage],
        generation: GenerationConfig,
    ) -> GenerationResult:
        """Generate one response for one benchmark example."""

    def metrics(self) -> dict[str, MetricValue]:
        """Return backend setup and hardware measurements."""


class FakeModelBackend:
    """Return deterministic fixture responses without a model or GPU."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses

    @property
    def name(self) -> str:
        return "fake"

    def generate(
        self,
        stable_id: str,
        messages: tuple[ChatMessage, ChatMessage],
        generation: GenerationConfig,
    ) -> GenerationResult:
        del messages, generation
        if stable_id not in self._responses:
            raise BackendError(f"fake backend has no response for {stable_id}")
        return GenerationResult(
            text=self._responses[stable_id],
            input_tokens=None,
            output_tokens=None,
        )

    def metrics(self) -> dict[str, MetricValue]:
        return {
            "backend_load_seconds": 0.0,
            "gpu_name": None,
            "peak_vram_allocated_bytes": None,
            "peak_vram_reserved_bytes": None,
        }


def require_cuda() -> None:
    """Fail before network access unless a CUDA-capable PyTorch runtime is active."""

    try:
        torch: Any = importlib.import_module("torch")
    except ImportError as error:
        raise BackendError(
            "CUDA smoke evaluation requires the pinned 'smoke' optional dependencies"
        ) from error
    if not torch.cuda.is_available():
        raise BackendError("CUDA is not available; refusing to download the model")


class HuggingFaceCudaBackend:
    """Pinned, greedy Transformers backend for the approved 10-example smoke run."""

    def __init__(self, config: EvaluationConfig) -> None:
        require_cuda()
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as error:
            raise BackendError(
                "real-model smoke evaluation requires the pinned 'smoke' dependencies"
            ) from error

        if config.model.device != "cuda" or config.model.dtype != "float16":
            raise BackendError("the approved smoke protocol requires CUDA float16")

        self._torch: Any = torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        self._tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            config.model.repo_id,
            revision=config.model.revision,
        )
        self._model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.repo_id,
            revision=config.model.revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self._model.to("cuda:0")
        self._model.eval()
        self._load_seconds = time.perf_counter() - load_started
        self._name = f"hf-cuda:{config.model.repo_id}@{config.model.revision}"

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
        message_dicts = [dict(message) for message in messages]
        input_ids: Any = self._tokenizer.apply_chat_template(
            message_dicts,
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
        text = cast(
            str,
            self._tokenizer.decode(generated_ids, skip_special_tokens=True),
        )
        return GenerationResult(
            text=text,
            input_tokens=prompt_tokens,
            output_tokens=int(generated_ids.shape[-1]),
        )

    def metrics(self) -> dict[str, MetricValue]:
        return {
            "backend_load_seconds": self._load_seconds,
            "gpu_name": str(self._torch.cuda.get_device_name(0)),
            "peak_vram_allocated_bytes": int(self._torch.cuda.max_memory_allocated(0)),
            "peak_vram_reserved_bytes": int(self._torch.cuda.max_memory_reserved(0)),
        }
