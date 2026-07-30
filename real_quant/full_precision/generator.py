from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .latency import LatencyRecord, cuda_synchronize_if_needed


def append_prompt_token(prompt: str, prompt_token: str | None) -> str:
    if prompt_token and not prompt.endswith(prompt_token):
        return prompt + prompt_token
    return prompt


def build_hf_generation_kwargs(
    *,
    max_new_tokens: int,
    num_beams: int | None,
    num_return_sequences: int,
    pad_token_id: int | None,
    eos_token_id: int | None,
    output_scores: bool = False,
    use_cache: bool = True,
) -> dict[str, Any]:
    if num_beams is not None and num_return_sequences > num_beams:
        raise ValueError(
            f"num_return_sequences ({num_return_sequences}) cannot be greater than num_beams ({num_beams})."
        )
    kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "num_return_sequences": int(num_return_sequences),
        "do_sample": False,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "use_cache": bool(use_cache),
    }
    if num_beams is not None:
        kwargs["num_beams"] = int(num_beams)
    if output_scores:
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
    return kwargs


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}.")
    return mapping[name]  # type: ignore[return-value]


def _extract_sequences(output: Any) -> torch.Tensor:
    return output.sequences if hasattr(output, "sequences") else output


def _progress_disabled_from_env() -> bool:
    return os.environ.get("OOR_DISABLE_TQDM", "").strip().lower() in {"1", "true", "yes", "on"}


def iter_batch_starts_with_progress(
    *,
    total_items: int,
    batch_size: int,
    desc: str,
):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    total_batches = (int(total_items) + int(batch_size) - 1) // int(batch_size)
    return tqdm(
        range(0, int(total_items), int(batch_size)),
        total=total_batches,
        desc=desc,
        unit="batch",
        disable=_progress_disabled_from_env(),
        dynamic_ncols=True,
    )


class HFFullPrecisionGenerator:
    """Single-process HuggingFace generator with OpenOneRec-style output timing."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        model_name: str,
        device: str | torch.device,
        num_params: float | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.device = torch.device(device)
        self.tensor_parallel_size = 1
        self.num_params = num_params
        self.latency_records: dict[str, LatencyRecord] = {}
        self.mfu_stats: dict[str, dict[str, list[float] | list[int]]] = {}

    def __str__(self) -> str:
        return os.path.basename(self.model_name.rstrip("/"))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: str | None = None,
    ) -> "HFFullPrecisionGenerator":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Left padding keeps generated tokens after the common padded prompt length in batched decoder-only generation.
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype_from_name(dtype),
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        model.to(device)
        model.eval()
        num_params = sum(p.numel() for p in model.parameters())
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_name=Path(model_path.rstrip("/")).name,
            device=device,
            num_params=float(num_params),
        )

    def generate_one(
        self,
        *,
        sample_id: str,
        prompt: str,
        generation_kwargs: Mapping[str, Any],
    ) -> tuple[list[str], LatencyRecord]:
        tokenize_start = time.perf_counter()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_len = int(inputs["input_ids"].shape[-1])
        tokenize_time = time.perf_counter() - tokenize_start

        cuda_synchronize_if_needed(self.device)
        generate_start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **dict(generation_kwargs))
        cuda_synchronize_if_needed(self.device)
        generate_time = time.perf_counter() - generate_start

        sequences = _extract_sequences(output).detach().cpu()
        decode_start = time.perf_counter()
        generations: list[str] = []
        generated_tokens = 0
        for seq in sequences:
            generated_ids = seq[prompt_len:]
            generated_tokens += int(generated_ids.numel())
            generations.append(self.tokenizer.decode(generated_ids, skip_special_tokens=False))
        decode_time = time.perf_counter() - decode_start

        record = LatencyRecord(
            sample_id=sample_id,
            prompt_tokens=prompt_len,
            generated_sequences=len(generations),
            generated_tokens=generated_tokens,
            tokenize_time=tokenize_time,
            generate_time=generate_time,
            decode_time=decode_time,
        )
        return generations, record

    def generate_batch(
        self,
        *,
        batch: Sequence[tuple[str, str]],
        generation_kwargs: Mapping[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, LatencyRecord]]:
        if not batch:
            return {}, {}
        sample_ids = [sample_id for sample_id, _prompt in batch]
        prompts = [prompt for _sample_id, prompt in batch]

        tokenize_start = time.perf_counter()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        prompt_batch_len = int(inputs["input_ids"].shape[-1])
        attention_mask = inputs.get("attention_mask")
        if torch.is_tensor(attention_mask):
            prompt_lengths = [int(v) for v in attention_mask.sum(dim=-1).detach().cpu().tolist()]
        else:
            prompt_lengths = [prompt_batch_len for _ in prompts]
        tokenize_time = time.perf_counter() - tokenize_start

        cuda_synchronize_if_needed(self.device)
        generate_start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **dict(generation_kwargs))
        cuda_synchronize_if_needed(self.device)
        generate_time = time.perf_counter() - generate_start

        sequences = _extract_sequences(output).detach().cpu()
        if sequences.shape[0] % len(batch) != 0:
            raise RuntimeError(
                f"Generated sequence count {sequences.shape[0]} is not divisible by batch size {len(batch)}."
            )
        sequences_per_sample = int(sequences.shape[0] // len(batch))

        decode_start = time.perf_counter()
        generations_by_id: dict[str, list[str]] = {}
        generated_tokens_by_id: dict[str, int] = {}
        for idx, sample_id in enumerate(sample_ids):
            sample_sequences = sequences[idx * sequences_per_sample : (idx + 1) * sequences_per_sample]
            sample_generations: list[str] = []
            sample_generated_tokens = 0
            for seq in sample_sequences:
                generated_ids = seq[prompt_batch_len:]
                sample_generated_tokens += int(generated_ids.numel())
                sample_generations.append(self.tokenizer.decode(generated_ids, skip_special_tokens=False))
            generations_by_id[sample_id] = sample_generations
            generated_tokens_by_id[sample_id] = sample_generated_tokens
        decode_time = time.perf_counter() - decode_start

        divisor = float(len(batch))
        records = {
            sample_id: LatencyRecord(
                sample_id=sample_id,
                prompt_tokens=prompt_lengths[idx],
                generated_sequences=len(generations_by_id[sample_id]),
                generated_tokens=generated_tokens_by_id[sample_id],
                tokenize_time=tokenize_time / divisor,
                generate_time=generate_time / divisor,
                decode_time=decode_time / divisor,
            )
            for idx, sample_id in enumerate(sample_ids)
        }
        return generations_by_id, records

    def generate(
        self,
        prompts: Mapping[str, str],
        **kwargs: Any,
    ) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
        prompt_token = kwargs.pop("prompt_token", None)
        batch_size = int(kwargs.pop("batch_size", 1))
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        output_scores = bool(kwargs.pop("output_scores", False))
        max_new_tokens = int(kwargs.pop("max_new_tokens", 3))
        num_beams = kwargs.pop("num_beams", 32)
        num_return_sequences = int(kwargs.pop("num_return_sequences", 32))
        generation_kwargs = build_hf_generation_kwargs(
            max_new_tokens=max_new_tokens,
            num_beams=None if num_beams is None else int(num_beams),
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_scores=output_scores,
        )

        generations: dict[str, list[str]] = {}
        self.latency_records = {}
        self.mfu_stats = {}
        formatted_items = [
            (sample_id, append_prompt_token(prompt, prompt_token))
            for sample_id, prompt in prompts.items()
        ]
        for start in iter_batch_starts_with_progress(
            total_items=len(formatted_items),
            batch_size=batch_size,
            desc=f"{self} generate",
        ):
            batch = formatted_items[start : start + batch_size]
            if batch_size == 1:
                sample_id, formatted_prompt = batch[0]
                sample_generations, record = self.generate_one(
                    sample_id=sample_id,
                    prompt=formatted_prompt,
                    generation_kwargs=generation_kwargs,
                )
                batch_generations = {sample_id: sample_generations}
                batch_records = {sample_id: record}
            else:
                batch_generations, batch_records = self.generate_batch(
                    batch=batch,
                    generation_kwargs=generation_kwargs,
                )
            generations.update(batch_generations)
            for sample_id, record in batch_records.items():
                self.latency_records[sample_id] = record
                self.mfu_stats[sample_id] = record.to_sample_fields()  # type: ignore[assignment]
        return generations, {}

    def get_hardware_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "gpu_count": 1,
            "tensor_parallel_size": 1,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "gpu_model": props.name,
                    "gpu_memory_total_gb": props.total_memory / 1024**3,
                }
            )
        return info
