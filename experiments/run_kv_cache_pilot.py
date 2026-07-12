#!/usr/bin/env python3
"""Pilot KV-cache compression experiment.

This script is intentionally small and reproducible. It is a smoke test for the
paper pipeline, not a definitive benchmark. It compares full KV replay with
simple sink/window cache-retention policies and fits a lightweight compression
risk model on step-level observations.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PastKV = Any


@dataclass(frozen=True)
class PromptExample:
    prompt_id: str
    text: str
    split: str
    answer: str = ""
    task_type: str = ""
    answer_position: str = ""
    context_words: int = 0


@dataclass(frozen=True)
class Policy:
    name: str
    sink: int
    window: int
    adaptive: bool = False


@dataclass(frozen=True)
class UCKV2Config:
    lambda_gate: float
    beta_salience: float
    evict_every: int
    recent_fraction: float
    min_recent: int
    prefill_query_tokens: int
    probe_layers: Tuple[int, ...]
    entropy_gate_low: float
    entropy_gate_high: float


@dataclass
class RiskModelBundle:
    status: str
    tolerance: float
    feature_cols: List[str]
    clf: Optional[Any] = None
    reason: str = ""
    calibration_length_min: int = 0
    calibration_length_max: int = 0
    calibration_length_margin: int = 0
    calibration_length_feature: str = "prompt_len"


@dataclass
class FullTrace:
    prompt_id: str
    prompt_text: str
    prompt_len: int
    generated_ids: List[int]
    generated_text: str
    decision_logits: List[torch.Tensor]
    next_logits: List[torch.Tensor]
    elapsed_s: float
    prefill_elapsed_s: float
    decode_elapsed_s: float
    peak_cache_bytes: int


class NumpyLogisticRegression:
    """Small deterministic logistic regressor for portable experiment images."""

    def __init__(self, l2: float = 1.0, max_iter: int = 100, tol: float = 1e-8):
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(value: np.ndarray) -> np.ndarray:
        value = np.clip(value, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-value))

    def fit(self, X: Any, y: Any) -> "NumpyLogisticRegression":
        values = np.asarray(X, dtype=np.float64)
        labels = np.asarray(y, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        self.scale_ = values.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        standardized = (values - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])

        positives = max(1, int(labels.sum()))
        negatives = max(1, int(len(labels) - labels.sum()))
        weights = np.where(
            labels == 1,
            len(labels) / (2.0 * positives),
            len(labels) / (2.0 * negatives),
        )
        weight_sum = float(weights.sum())
        regularization = self.l2 / weight_sum
        coef = np.zeros(design.shape[1], dtype=np.float64)

        for _ in range(self.max_iter):
            probs = self._sigmoid(design @ coef)
            gradient = design.T @ (weights * (probs - labels)) / weight_sum
            gradient[1:] += regularization * coef[1:]
            curvature = weights * probs * (1.0 - probs)
            hessian = design.T @ (design * curvature[:, None]) / weight_sum
            hessian[1:, 1:] += regularization * np.eye(design.shape[1] - 1)
            hessian += 1e-9 * np.eye(design.shape[1])
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            coef -= step
            if float(np.linalg.norm(step)) <= self.tol:
                break

        self.coef_ = coef
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.coef_ is None:
            raise RuntimeError("Risk model must be fit before prediction.")
        values = np.asarray(X, dtype=np.float64)
        standardized = (values - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(standardized)), standardized])
        positive = self._sigmoid(design @ self.coef_)
        return np.column_stack([1.0 - positive, positive])


def brier_score_loss(y_true: Any, y_prob: Any) -> float:
    labels = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((labels - probs) ** 2))


def binary_log_loss(y_true: Any, y_prob: Any) -> float:
    labels = np.asarray(y_true, dtype=np.float64)
    probs = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


def binary_roc_auc(y_true: Any, y_prob: Any) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    ranks = np.empty(len(probs), dtype=np.float64)
    start = 0
    while start < len(sorted_probs):
        end = start + 1
        while end < len(sorted_probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilgpt2", help="HF model id or local path.")
    parser.add_argument("--prompts", default="experiments/prompts/pilot_prompts.jsonl")
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--task-filter", default="")
    parser.add_argument("--answer-position-filter", default="")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--split-mode", default="first-half", choices=["first-half", "alternating"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Wrap each prompt as a user message with the tokenizer chat template.",
    )
    parser.add_argument("--kl-risk-threshold", type=float, default=0.05)
    parser.add_argument("--uckv-risk-tolerance", type=float, default=0.5)
    parser.add_argument("--uckv-candidate-windows", default="4,8,16,32")
    parser.add_argument(
        "--keynorm-heavy-hitter-budgets",
        default="",
        help="Optional total token budgets for a deployable key-norm importance baseline.",
    )
    parser.add_argument(
        "--h2o-heavy-hitter-budgets",
        default="",
        help="Optional total budgets for an online cumulative-attention H2O-style baseline.",
    )
    parser.add_argument(
        "--uckv2-probe-layers",
        default="",
        help=(
            "Optional comma-separated 0-indexed attention layers for UCKV-2 "
            "utility aggregation. Empty means all returned attention layers."
        ),
    )
    parser.add_argument(
        "--uckv2-lambda",
        type=float,
        default=1.0,
        help="Entropy-gate multiplier for UCKV-2 fixed-budget scoring.",
    )
    parser.add_argument(
        "--uckv2-beta",
        type=float,
        default=0.0,
        help="Prefill salience multiplier for UCKV-2 fixed-budget scoring.",
    )
    parser.add_argument(
        "--uckv2-evict-every",
        type=int,
        default=16,
        help="Run UCKV-2 cache eviction every K decode steps.",
    )
    parser.add_argument(
        "--uckv2-recent-fraction",
        type=float,
        default=0.25,
        help="Recent-token reservation as a fraction of the fixed budget.",
    )
    parser.add_argument(
        "--uckv2-min-recent",
        type=int,
        default=64,
        help="Minimum recent-token reservation for UCKV-2 fixed-budget policies.",
    )
    parser.add_argument(
        "--uckv2-prefill-query-tokens",
        type=int,
        default=16,
        help="Number of final prompt tokens used to compute prefill salience.",
    )
    parser.add_argument(
        "--enable-length-guard",
        action="store_true",
        help="Add a UCKV variant that abstains to full cache outside calibration lengths.",
    )
    parser.add_argument("--free-run-eval", action="store_true")
    parser.add_argument(
        "--free-run-policies",
        default="full,uckv_budget",
        help=(
            "Comma-separated policies. h2o_matched_B reuses the UCKV-2 recent "
            "window, probe layers, and eviction cadence for a controlled utility "
            "comparison; h2o_hh_B retains the legacy H2O-style contract."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N prompts; set to 0 to disable.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but not available.")
    return device


def select_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def parse_windows(value: str) -> List[int]:
    windows = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("--uckv-candidate-windows must contain positive integers.")
    return windows


def parse_optional_indices(value: str) -> Tuple[int, ...]:
    indices = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(index < 0 for index in indices):
        raise ValueError("Layer indices must be non-negative.")
    return indices


def parse_csv_names(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def matches_filter(value: str, allowed: Sequence[str]) -> bool:
    return not allowed or value in allowed


def sink_window_policy_name(window: int) -> str:
    return f"sink4_window_{window}"


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_")


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def answer_in_text(answer: str, generated: str) -> bool:
    if not answer:
        return False
    answer_norm = normalize_text(answer)
    generated_norm = normalize_text(generated)
    if not answer_norm:
        return False
    return answer_norm in generated_norm


def safe_nanmean(values: Sequence[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return np.nan
    return float(np.mean(finite))


def load_prompts(
    path: Path,
    max_prompts: int,
    split_mode: str,
    task_filter: Sequence[str],
    answer_position_filter: Sequence[str],
) -> List[PromptExample]:
    examples: List[PromptExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task_type = row.get("task_type", row.get("task", ""))
            answer_position = row.get("answer_position", row.get("position", ""))
            if not matches_filter(task_type, task_filter):
                continue
            if not matches_filter(answer_position, answer_position_filter):
                continue
            examples.append(
                PromptExample(
                    prompt_id=row["id"],
                    text=row["text"],
                    split="",
                    answer=row.get("answer", ""),
                    task_type=task_type,
                    answer_position=answer_position,
                    context_words=int(row.get("context_words", 0) or 0),
                )
            )
            if len(examples) >= max_prompts:
                break
    if len(examples) < 2:
        raise ValueError("Need at least two prompts for calibration/evaluation split.")
    if split_mode == "first-half":
        split_at = max(1, len(examples) // 2)
        return [
            PromptExample(
                ex.prompt_id,
                ex.text,
                "calibration" if i < split_at else "evaluation",
                ex.answer,
                ex.task_type,
                ex.answer_position,
                ex.context_words,
            )
            for i, ex in enumerate(examples)
        ]
    if split_mode == "alternating":
        return [
            PromptExample(
                ex.prompt_id,
                ex.text,
                "calibration" if i % 2 == 0 else "evaluation",
                ex.answer,
                ex.task_type,
                ex.answer_position,
                ex.context_words,
            )
            for i, ex in enumerate(examples)
        ]
    raise ValueError(f"Unknown split mode: {split_mode}")


def apply_chat_template_to_prompts(
    prompts: Sequence[PromptExample],
    tokenizer: Any,
) -> List[PromptExample]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("--apply-chat-template requires a tokenizer with chat_template.")
    templated: List[PromptExample] = []
    for example in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        templated.append(
            PromptExample(
                prompt_id=example.prompt_id,
                text=text,
                split=example.split,
                answer=example.answer,
                task_type=example.task_type,
                answer_position=example.answer_position,
                context_words=example.context_words,
            )
        )
    return templated


def cache_items(past: PastKV) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(past, "layers"):
        return tuple((layer.keys, layer.values) for layer in past.layers)
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return tuple(zip(past.key_cache, past.value_cache))
    return tuple((layer[0], layer[1]) for layer in past)


def cache_len(past: PastKV) -> int:
    return int(cache_items(past)[0][0].shape[-2])


def cache_nbytes(past: PastKV) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for layer in cache_items(past)
        for tensor in layer
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def slice_past(past: PastKV, keep_indices: Sequence[int]) -> PastKV:
    if len(keep_indices) == cache_len(past):
        return past
    layers = cache_items(past)
    device = layers[0][0].device
    idx = torch.tensor(list(keep_indices), dtype=torch.long, device=device)
    if hasattr(past, "layers"):
        for layer in past.layers:
            layer.keys = layer.keys.index_select(-2, idx)
            layer.values = layer.values.index_select(-2, idx)
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = len(keep_indices)
        return past
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for layer_idx, (key, value) in enumerate(layers):
            past.key_cache[layer_idx] = key.index_select(-2, idx)
            past.value_cache[layer_idx] = value.index_select(-2, idx)
        if hasattr(past, "_seen_tokens"):
            past._seen_tokens = len(keep_indices)
        return past
    sliced: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for key, value in layers:
        sliced.append((key.index_select(-2, idx), value.index_select(-2, idx)))
    return tuple(sliced)


def select_keep_indices(
    positions: Sequence[int],
    sink: int,
    window: int,
    current_abs_pos: int,
) -> List[int]:
    keep: List[int] = []
    recent_start = max(0, current_abs_pos - window)
    for idx, pos in enumerate(positions):
        if pos < sink or pos >= recent_start:
            keep.append(idx)
    if not keep:
        keep = [len(positions) - 1]
    return sorted(set(keep))


def select_keynorm_heavy_hitter_indices(
    past: PastKV,
    positions: Sequence[int],
    sink: int,
    budget: int,
    current_abs_pos: int,
) -> List[int]:
    """Keep sinks, recent tokens, and high-norm keys under one shared budget."""
    if len(positions) <= budget:
        return list(range(len(positions)))
    recent_window = max(1, budget // 2)
    recent_start = max(0, current_abs_pos - recent_window)
    mandatory = {
        idx for idx, pos in enumerate(positions) if pos < sink or pos >= recent_start
    }
    if len(mandatory) >= budget:
        return sorted(mandatory, key=lambda idx: positions[idx])[-budget:]

    layers = cache_items(past)
    scores = torch.zeros(len(positions), dtype=torch.float32, device=layers[0][0].device)
    for key, _ in layers:
        scores += key.detach().float().square().mean(dim=(0, 1, 3))
    if mandatory:
        mandatory_idx = torch.tensor(sorted(mandatory), dtype=torch.long, device=scores.device)
        scores.index_fill_(0, mandatory_idx, float("-inf"))
    heavy_count = min(budget - len(mandatory), len(positions) - len(mandatory))
    heavy = torch.topk(scores, k=heavy_count, largest=True).indices.tolist()
    return sorted(mandatory.union(int(idx) for idx in heavy))


def select_score_heavy_hitter_indices(
    scores: torch.Tensor,
    positions: Sequence[int],
    sink: int,
    budget: int,
    current_abs_pos: int,
    min_recent: int = 1,
    recent_fraction: float = 0.5,
) -> List[int]:
    if len(positions) <= budget:
        return list(range(len(positions)))
    recent_window = max(1, min_recent, int(round(budget * recent_fraction)))
    recent_start = max(0, current_abs_pos - recent_window)
    mandatory = {
        idx for idx, pos in enumerate(positions) if pos < sink or pos >= recent_start
    }
    if len(mandatory) >= budget:
        return sorted(mandatory, key=lambda idx: positions[idx])[-budget:]
    selectable = scores.detach().float().clone()
    if mandatory:
        mandatory_idx = torch.tensor(
            sorted(mandatory), dtype=torch.long, device=selectable.device
        )
        selectable.index_fill_(0, mandatory_idx, float("-inf"))
    heavy_count = min(budget - len(mandatory), len(positions) - len(mandatory))
    heavy = torch.topk(selectable, k=heavy_count, largest=True).indices.tolist()
    return sorted(mandatory.union(int(idx) for idx in heavy))


def select_uckv2_heavy_hitter_indices(
    scores: torch.Tensor,
    positions: Sequence[int],
    sink: int,
    budget: int,
    current_abs_pos: int,
    min_recent: int,
    recent_fraction: float,
) -> List[int]:
    return select_score_heavy_hitter_indices(
        scores,
        positions,
        sink,
        budget,
        current_abs_pos,
        min_recent=min_recent,
        recent_fraction=recent_fraction,
    )


def selected_attention_layers(
    attentions: Sequence[Optional[torch.Tensor]],
    probe_layers: Sequence[int],
) -> List[torch.Tensor]:
    if probe_layers:
        invalid = [index for index in probe_layers if index >= len(attentions)]
        if invalid:
            raise ValueError(
                f"Probe layer index out of range: {invalid}; model returned "
                f"{len(attentions)} attention tensors."
            )
        selected = [attentions[index] for index in probe_layers]
    else:
        selected = list(attentions)
    return [attention for attention in selected if attention is not None]


def aggregate_attention_scores(
    attentions: Sequence[Optional[torch.Tensor]],
    probe_layers: Sequence[int] = (),
) -> torch.Tensor:
    layer_scores = [
        attention.detach().float().mean(dim=(0, 1, 2))
        for attention in selected_attention_layers(attentions, probe_layers)
    ]
    if not layer_scores:
        raise RuntimeError("The H2O-style baseline requires eager attention outputs.")
    return torch.stack(layer_scores).mean(dim=0)


def prefill_salience_from_attentions(
    attentions: Sequence[Optional[torch.Tensor]],
    prompt_len: int,
    query_tokens: int,
    probe_layers: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    layers = selected_attention_layers(attentions, probe_layers)
    if not layers:
        return torch.zeros(prompt_len, dtype=torch.float32, device=device)
    query_start = max(0, prompt_len - max(1, query_tokens))
    salience_by_layer: List[torch.Tensor] = []
    for attention in layers:
        prompt_attention = attention.detach().float()[:, :, query_start:prompt_len, :prompt_len]
        salience_by_layer.append(prompt_attention.mean(dim=(0, 1, 2)))
    return torch.stack(salience_by_layer).mean(dim=0).to(device)


def tensor_entropy_and_margin(logits: torch.Tensor) -> Tuple[float, float, float]:
    logits_cpu = logits.detach().float().cpu()
    probs = torch.softmax(logits_cpu, dim=-1)
    log_probs = torch.log_softmax(logits_cpu, dim=-1)
    entropy = float(-(probs * log_probs).sum().item())
    vocab = logits_cpu.shape[-1]
    normalized_entropy = entropy / math.log(vocab)
    top2 = torch.topk(probs, k=2, dim=-1).values.squeeze(0)
    margin = float((top2[0] - top2[1]).item())
    return entropy, normalized_entropy, margin


def entropy_gate(norm_entropy: float, low_q: float, high_q: float) -> float:
    denom = max(1e-8, high_q - low_q)
    return float(np.clip((norm_entropy - low_q) / denom, 0.0, 1.0))


def kl_full_to_compressed(full_logits: torch.Tensor, compressed_logits: torch.Tensor) -> float:
    full = full_logits.detach().float().cpu()
    comp = compressed_logits.detach().float().cpu()
    full_log_probs = torch.log_softmax(full, dim=-1)
    comp_log_probs = torch.log_softmax(comp, dim=-1)
    full_probs = torch.softmax(full, dim=-1)
    kl = float((full_probs * (full_log_probs - comp_log_probs)).sum().item())
    return max(0.0, kl)


def logits_limit_kwargs(model: Any) -> Dict[str, int]:
    if hasattr(model, "_uckv_logits_limit_kwargs"):
        return model._uckv_logits_limit_kwargs
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in parameters:
        kwargs = {"logits_to_keep": 1}
    elif "num_logits_to_keep" in parameters:
        kwargs = {"num_logits_to_keep": 1}
    else:
        kwargs = {}
    model._uckv_logits_limit_kwargs = kwargs
    return kwargs


def forward_prefill(model: Any, input_ids: torch.Tensor) -> Tuple[PastKV, torch.Tensor]:
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
        **logits_limit_kwargs(model),
    )
    return out.past_key_values, out.logits[:, -1, :]


def forward_prefill_with_attentions(
    model: Any,
    input_ids: torch.Tensor,
) -> Tuple[PastKV, torch.Tensor, Sequence[Optional[torch.Tensor]]]:
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
        **logits_limit_kwargs(model),
    )
    if out.attentions is None:
        raise RuntimeError("Eager prefill did not return attention weights.")
    return out.past_key_values, out.logits[:, -1, :], out.attentions


def forward_one(
    model: Any,
    token_id: int,
    past: PastKV,
    past_positions: Sequence[int],
    abs_pos: int,
    device: torch.device,
) -> Tuple[PastKV, torch.Tensor]:
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(past_positions) + 1), dtype=torch.long, device=device)
    position_ids = torch.tensor([[abs_pos]], dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past,
        use_cache=True,
        return_dict=True,
        **logits_limit_kwargs(model),
    )
    return out.past_key_values, out.logits[:, -1, :]


@torch.no_grad()
def build_full_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: PromptExample,
    max_new_tokens: int,
) -> FullTrace:
    encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    synchronize_device(device)
    start = time.perf_counter()
    past, logits = forward_prefill(model, input_ids)
    synchronize_device(device)
    prefill_elapsed_s = time.perf_counter() - start
    prompt_len = int(input_ids.shape[1])
    positions = list(range(prompt_len))
    generated: List[int] = []
    decision_logits: List[torch.Tensor] = []
    next_logits: List[torch.Tensor] = []
    current_logits = logits
    peak_cache_bytes = cache_nbytes(past)
    decode_start = time.perf_counter()

    for step in range(max_new_tokens):
        decision_logits.append(current_logits.detach().cpu())
        next_token = int(torch.argmax(current_logits, dim=-1).item())
        generated.append(next_token)
        abs_pos = prompt_len + step
        past, current_logits = forward_one(model, next_token, past, positions, abs_pos, device)
        positions.append(abs_pos)
        next_logits.append(current_logits.detach().cpu())
        peak_cache_bytes = max(peak_cache_bytes, cache_nbytes(past))

    synchronize_device(device)
    decode_elapsed_s = time.perf_counter() - decode_start
    elapsed_s = prefill_elapsed_s + decode_elapsed_s
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return FullTrace(
        prompt_id=example.prompt_id,
        prompt_text=example.text,
        prompt_len=prompt_len,
        generated_ids=generated,
        generated_text=generated_text,
        decision_logits=decision_logits,
        next_logits=next_logits,
        elapsed_s=elapsed_s,
        prefill_elapsed_s=prefill_elapsed_s,
        decode_elapsed_s=decode_elapsed_s,
        peak_cache_bytes=peak_cache_bytes,
    )


def adaptive_window(norm_entropy: float, low_q: float, high_q: float) -> int:
    if norm_entropy >= high_q:
        return 32
    if norm_entropy >= low_q:
        return 16
    return 8


@torch.no_grad()
def replay_policy(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: PromptExample,
    trace: FullTrace,
    policy: Policy,
    entropy_low_q: float,
    entropy_high_q: float,
) -> List[Dict[str, Any]]:
    encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    past, _ = forward_prefill(model, input_ids)
    positions = list(range(trace.prompt_len))
    rows: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for step, token_id in enumerate(trace.generated_ids):
        current_abs_pos = trace.prompt_len + step
        decision_logits = trace.decision_logits[step]
        entropy, norm_entropy, margin = tensor_entropy_and_margin(decision_logits)

        if policy.name == "full":
            keep_indices = list(range(len(positions)))
            applied_sink = len(positions)
            applied_window = len(positions)
        elif policy.name.startswith("keynorm_hh_"):
            applied_sink = policy.sink
            applied_window = policy.window
            keep_indices = select_keynorm_heavy_hitter_indices(
                past,
                positions,
                sink=applied_sink,
                budget=applied_window,
                current_abs_pos=current_abs_pos,
            )
        else:
            applied_sink = policy.sink
            applied_window = (
                adaptive_window(norm_entropy, entropy_low_q, entropy_high_q)
                if policy.adaptive
                else policy.window
            )
            keep_indices = select_keep_indices(
                positions,
                sink=applied_sink,
                window=applied_window,
                current_abs_pos=current_abs_pos,
            )

        past = slice_past(past, keep_indices)
        positions = [positions[i] for i in keep_indices]

        kept_tokens_before = len(positions)
        full_tokens_before = current_abs_pos
        retained_ratio = kept_tokens_before / max(1, full_tokens_before)

        past, compressed_logits = forward_one(model, token_id, past, positions, current_abs_pos, device)
        positions.append(current_abs_pos)
        full_next_logits = trace.next_logits[step]
        kl = kl_full_to_compressed(full_next_logits, compressed_logits.detach().cpu())
        full_top1 = int(torch.argmax(full_next_logits, dim=-1).item())
        compressed_top1 = int(torch.argmax(compressed_logits.detach().cpu(), dim=-1).item())
        rows.append(
            {
                "prompt_id": example.prompt_id,
                "split": example.split,
                "policy": policy.name,
                "step": step,
                "prompt_len": trace.prompt_len,
                "context_words": example.context_words,
                "full_tokens_before": full_tokens_before,
                "kept_tokens_before": kept_tokens_before,
                "retained_ratio": retained_ratio,
                "sink": applied_sink,
                "window": applied_window,
                "entropy": entropy,
                "norm_entropy": norm_entropy,
                "margin": margin,
                "kl_full_to_compressed": kl,
                "top1_mismatch": int(full_top1 != compressed_top1),
                "full_next_top1": full_top1,
                "compressed_next_top1": compressed_top1,
            }
        )

    elapsed = time.perf_counter() - start
    for row in rows:
        row["replay_elapsed_s_total"] = elapsed
        row["replay_elapsed_s_per_step"] = elapsed / max(1, len(rows))
    return rows


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(conf - acc)
    return float(ece)


def build_risk_features(rows: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    return rows.loc[:, list(feature_cols)].astype(float)


def fit_risk_model(
    steps: pd.DataFrame,
    kl_threshold: float,
    risk_tolerance: float,
    training_policies: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, RiskModelBundle]:
    training_policy_set = set(training_policies)
    risk_rows = steps[steps["policy"].isin(training_policy_set)].copy()
    risk_rows["risk_label"] = (
        (risk_rows["top1_mismatch"] == 1)
        | (risk_rows["kl_full_to_compressed"] > kl_threshold)
    ).astype(int)

    feature_cols = [
        "norm_entropy",
        "margin",
        "retained_ratio",
        "kept_tokens_before",
        "full_tokens_before",
        "sink",
        "window",
    ]
    X = build_risk_features(risk_rows, feature_cols)
    y = risk_rows["risk_label"].to_numpy()
    train_mask = risk_rows["split"].eq("calibration").to_numpy()
    eval_mask = risk_rows["split"].eq("evaluation").to_numpy()
    calibration_length_feature = "prompt_len"
    calibration_lengths = risk_rows.loc[train_mask, calibration_length_feature].astype(int)
    calibration_length_min = int(calibration_lengths.min())
    calibration_length_max = int(calibration_lengths.max())
    calibration_length_margin = max(32, int(round(0.05 * calibration_length_max)))

    predictions = risk_rows[["prompt_id", "split", "policy", "step", "risk_label"]].copy()
    predictions["risk_pred"] = np.nan
    summary_rows: List[Dict[str, Any]] = []

    if train_mask.sum() < 4 or eval_mask.sum() < 4 or len(np.unique(y[train_mask])) < 2:
        reason = "Need two risk classes and enough calibration/evaluation rows in sink4_window candidates."
        training_policy_text = ",".join(sorted(training_policy_set))
        summary_rows.append(
            {
                "model": "logistic_risk",
                "status": "skipped",
                "reason": reason,
                "kl_risk_threshold": kl_threshold,
                "uckv_risk_tolerance": risk_tolerance,
                "training_policies": training_policy_text,
                "calibration_length_feature": calibration_length_feature,
                "calibration_length_min": calibration_length_min,
                "calibration_length_max": calibration_length_max,
                "calibration_length_margin": calibration_length_margin,
            }
        )
        bundle = RiskModelBundle(
            status="skipped",
            tolerance=risk_tolerance,
            feature_cols=feature_cols,
            reason=reason,
            calibration_length_min=calibration_length_min,
            calibration_length_max=calibration_length_max,
            calibration_length_margin=calibration_length_margin,
            calibration_length_feature=calibration_length_feature,
        )
        return predictions, pd.DataFrame(summary_rows), bundle

    clf = NumpyLogisticRegression(max_iter=100)
    clf.fit(X.loc[train_mask], y[train_mask])
    risk_pred = clf.predict_proba(X)[:, 1]
    predictions["risk_pred"] = risk_pred

    for split_name, mask in [("calibration", train_mask), ("evaluation", eval_mask)]:
        y_split = y[mask]
        p_split = risk_pred[mask]
        training_policy_text = ",".join(sorted(training_policy_set))
        row: Dict[str, Any] = {
            "model": "logistic_risk",
            "status": "ok",
            "split": split_name,
            "n": int(mask.sum()),
            "positive_rate": float(np.mean(y_split)),
            "brier": float(brier_score_loss(y_split, p_split)),
            "ece_10": expected_calibration_error(y_split, p_split, bins=10),
            "kl_risk_threshold": kl_threshold,
            "uckv_risk_tolerance": risk_tolerance,
            "training_policies": training_policy_text,
            "calibration_length_feature": calibration_length_feature,
            "calibration_length_min": calibration_length_min,
            "calibration_length_max": calibration_length_max,
            "calibration_length_margin": calibration_length_margin,
        }
        if len(np.unique(y_split)) == 2:
            row["roc_auc"] = binary_roc_auc(y_split, p_split)
            row["log_loss"] = binary_log_loss(y_split, p_split)
        else:
            row["roc_auc"] = np.nan
            row["log_loss"] = np.nan
        summary_rows.append(row)

    bundle = RiskModelBundle(
        status="ok",
        tolerance=risk_tolerance,
        feature_cols=feature_cols,
        clf=clf,
        calibration_length_min=calibration_length_min,
        calibration_length_max=calibration_length_max,
        calibration_length_margin=calibration_length_margin,
        calibration_length_feature=calibration_length_feature,
    )
    return predictions, pd.DataFrame(summary_rows), bundle


def predict_risk(bundle: RiskModelBundle, candidate_rows: pd.DataFrame) -> np.ndarray:
    if bundle.status != "ok" or bundle.clf is None:
        raise RuntimeError(f"Risk model is unavailable: {bundle.reason}")
    X = build_risk_features(candidate_rows, bundle.feature_cols)
    return bundle.clf.predict_proba(X)[:, 1]


def select_uckv_keep_indices(
    positions: Sequence[int],
    current_abs_pos: int,
    logits: torch.Tensor,
    risk_bundle: RiskModelBundle,
    candidate_windows: Sequence[int],
    sink: int,
) -> Tuple[List[int], int, float, str]:
    _, norm_entropy, margin = tensor_entropy_and_margin(logits)
    full_tokens_before = current_abs_pos
    candidate_records: List[Dict[str, Any]] = []
    for window in candidate_windows:
        keep_indices = select_keep_indices(
            positions,
            sink=sink,
            window=window,
            current_abs_pos=current_abs_pos,
        )
        kept_tokens = len(keep_indices)
        candidate_records.append(
            {
                "sink": sink,
                "window": window,
                "norm_entropy": norm_entropy,
                "margin": margin,
                "retained_ratio": kept_tokens / max(1, full_tokens_before),
                "kept_tokens_before": kept_tokens,
                "full_tokens_before": full_tokens_before,
                "keep_indices": keep_indices,
            }
        )
    candidates = pd.DataFrame(candidate_records)
    risks = predict_risk(risk_bundle, candidates)
    candidates["pred_risk"] = risks
    safe = candidates[candidates["pred_risk"] <= risk_bundle.tolerance].copy()
    if len(safe):
        selected = safe.sort_values(["kept_tokens_before", "window"]).iloc[0]
        selection_reason = "min_safe_budget"
    else:
        selected = candidates.sort_values(["kept_tokens_before", "window"]).iloc[-1]
        selection_reason = "fallback_max_budget"
    return (
        list(selected["keep_indices"]),
        int(selected["window"]),
        float(selected["pred_risk"]),
        selection_reason,
    )


@torch.no_grad()
def replay_uckv_budget(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: PromptExample,
    trace: FullTrace,
    risk_bundle: RiskModelBundle,
    candidate_windows: Sequence[int],
    sink: int = 4,
    length_guard: bool = False,
) -> List[Dict[str, Any]]:
    encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    past, _ = forward_prefill(model, input_ids)
    positions = list(range(trace.prompt_len))
    rows: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for step, token_id in enumerate(trace.generated_ids):
        current_abs_pos = trace.prompt_len + step
        decision_logits = trace.decision_logits[step]
        entropy, norm_entropy, margin = tensor_entropy_and_margin(decision_logits)
        full_tokens_before = current_abs_pos

        observed_length = trace.prompt_len
        outside_calibration = not (
            risk_bundle.calibration_length_min - risk_bundle.calibration_length_margin
            <= observed_length
            <= risk_bundle.calibration_length_max + risk_bundle.calibration_length_margin
        )
        if length_guard and outside_calibration:
            keep_indices = list(range(len(positions)))
            selected_window = len(positions)
            selected_risk = float("nan")
            selection_reason = "length_shift_abstain_full"
        else:
            keep_indices, selected_window, selected_risk, selection_reason = (
                select_uckv_keep_indices(
                    positions,
                    current_abs_pos,
                    decision_logits,
                    risk_bundle,
                    candidate_windows,
                    sink,
                )
            )
        past = slice_past(past, keep_indices)
        positions = [positions[i] for i in keep_indices]

        kept_tokens_before = len(positions)
        retained_ratio = kept_tokens_before / max(1, full_tokens_before)
        past, compressed_logits = forward_one(model, token_id, past, positions, current_abs_pos, device)
        positions.append(current_abs_pos)

        full_next_logits = trace.next_logits[step]
        kl = kl_full_to_compressed(full_next_logits, compressed_logits.detach().cpu())
        full_top1 = int(torch.argmax(full_next_logits, dim=-1).item())
        compressed_top1 = int(torch.argmax(compressed_logits.detach().cpu(), dim=-1).item())
        rows.append(
            {
                "prompt_id": example.prompt_id,
                "split": example.split,
                "policy": "uckv_length_guard" if length_guard else "uckv_budget",
                "step": step,
                "prompt_len": trace.prompt_len,
                "context_words": example.context_words,
                "full_tokens_before": full_tokens_before,
                "kept_tokens_before": kept_tokens_before,
                "retained_ratio": retained_ratio,
                "sink": sink,
                "window": selected_window,
                "entropy": entropy,
                "norm_entropy": norm_entropy,
                "margin": margin,
                "kl_full_to_compressed": kl,
                "top1_mismatch": int(full_top1 != compressed_top1),
                "full_next_top1": full_top1,
                "compressed_next_top1": compressed_top1,
                "uckv_pred_risk": selected_risk,
                "uckv_risk_tolerance": risk_bundle.tolerance,
                "uckv_selection": selection_reason,
            }
        )

    elapsed = time.perf_counter() - start
    for row in rows:
        row["replay_elapsed_s_total"] = elapsed
        row["replay_elapsed_s_per_step"] = elapsed / max(1, len(rows))
    return rows


def parse_policy_name(name: str) -> Policy:
    if name == "full":
        return Policy("full", sink=0, window=0)
    match = re.fullmatch(r"sink(\d+)_window_(\d+)", name)
    if match:
        return Policy(name, sink=int(match.group(1)), window=int(match.group(2)))
    match = re.fullmatch(r"keynorm_hh_(\d+)", name)
    if match:
        return Policy(name, sink=4, window=int(match.group(1)))
    raise ValueError(f"Unsupported free-run policy: {name}")


@torch.no_grad()
def free_run_h2o_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: PromptExample,
    budget: int,
    max_new_tokens: int,
    sink: int = 4,
    evict_every: int = 1,
    recent_fraction: float = 0.5,
    min_recent: int = 1,
    probe_layers: Sequence[int] = (),
    policy_label: Optional[str] = None,
) -> Dict[str, Any]:
    encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    cuda_baseline_bytes = 0
    if device.type == "cuda":
        cuda_baseline_bytes = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_device(device)
    prefill_start = time.perf_counter()
    past, current_logits = forward_prefill(model, input_ids)
    synchronize_device(device)
    prefill_elapsed_s = time.perf_counter() - prefill_start

    prompt_len = int(input_ids.shape[1])
    positions = list(range(prompt_len))
    scores = torch.zeros(prompt_len, dtype=torch.float32, device=device)
    generated: List[int] = []
    kept_counts: List[int] = []
    budget_overruns: List[int] = []
    cache_bytes: List[int] = []
    eviction_count = 0
    peak_cache_bytes = cache_nbytes(past)
    can_set_attention = hasattr(model, "set_attn_implementation")
    original_implementation = str(getattr(model.config, "_attn_implementation", ""))
    if can_set_attention:
        model.set_attn_implementation("eager")
    decode_start = time.perf_counter()
    try:
        for step in range(max_new_tokens):
            abs_pos = prompt_len + step
            next_token = int(torch.argmax(current_logits, dim=-1).item())
            step_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            attention_mask = torch.ones(
                (1, len(positions) + 1), dtype=torch.long, device=device
            )
            position_ids = torch.tensor([[abs_pos]], dtype=torch.long, device=device)
            output = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                **logits_limit_kwargs(model),
            )
            if output.attentions is None:
                raise RuntimeError("Eager attention did not return attention weights.")
            step_scores = aggregate_attention_scores(
                output.attentions,
                probe_layers=probe_layers,
            ).to(scores.device)
            if len(step_scores) != len(positions) + 1:
                raise RuntimeError(
                    "Attention/cache length mismatch in H2O-style baseline: "
                    f"attention={len(step_scores)} cache={len(positions) + 1}."
                )
            scores = torch.cat([scores + step_scores[:-1], step_scores[-1:]])
            past = output.past_key_values
            current_logits = output.logits[:, -1, :]
            positions.append(abs_pos)
            generated.append(next_token)
            peak_cache_bytes = max(peak_cache_bytes, cache_nbytes(past))

            should_evict = step == 0 or (step + 1) % evict_every == 0
            if should_evict:
                keep = select_score_heavy_hitter_indices(
                    scores,
                    positions,
                    sink=sink,
                    budget=budget,
                    current_abs_pos=abs_pos,
                    min_recent=min_recent,
                    recent_fraction=recent_fraction,
                )
                past = slice_past(past, keep)
                positions = [positions[index] for index in keep]
                keep_tensor = torch.tensor(keep, dtype=torch.long, device=scores.device)
                scores = scores.index_select(0, keep_tensor)
                eviction_count += 1
            kept_counts.append(len(positions))
            budget_overruns.append(max(0, len(positions) - budget))
            cache_bytes.append(cache_nbytes(past))
    finally:
        if can_set_attention:
            model.set_attn_implementation(original_implementation)

    synchronize_device(device)
    decode_elapsed_s = time.perf_counter() - decode_start
    elapsed_s = prefill_elapsed_s + decode_elapsed_s
    cuda_peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt_id": example.prompt_id,
        "split": example.split,
        "task_type": example.task_type,
        "answer_position": example.answer_position,
        "context_words": example.context_words,
        "policy": policy_label or f"h2o_hh_{budget}",
        "expected_answer": example.answer,
        "answer_contains": int(answer_in_text(example.answer, generated_text)),
        "generated_tokens": len(generated),
        "generated_text": generated_text,
        "elapsed_s": elapsed_s,
        "prefill_elapsed_s": prefill_elapsed_s,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tokens_per_s": (
            len(generated) / decode_elapsed_s if decode_elapsed_s > 0 else np.nan
        ),
        "peak_cache_bytes": peak_cache_bytes,
        "avg_cache_bytes": float(np.mean(cache_bytes)) if cache_bytes else np.nan,
        "cuda_peak_allocated_bytes": cuda_peak_bytes,
        "cuda_peak_delta_bytes": max(0, cuda_peak_bytes - cuda_baseline_bytes),
        "avg_kept_tokens": float(np.mean(kept_counts)) if kept_counts else np.nan,
        "max_kept_tokens": max(kept_counts) if kept_counts else np.nan,
        "budget_overrun_steps": sum(value > 0 for value in budget_overruns),
        "max_budget_overrun": max(budget_overruns) if budget_overruns else 0,
        "avg_selected_window": float(budget),
        "avg_selected_risk": np.nan,
        "fallback_steps": 0,
        "abstain_steps": 0,
        "selector_elapsed_s": np.nan,
        "attention_score_elapsed_s": np.nan,
        "utility_elapsed_s": np.nan,
        "topk_elapsed_s": np.nan,
        "slice_elapsed_s": np.nan,
        "selector_decode_fraction": np.nan,
        "eviction_count": eviction_count,
        "evict_every": evict_every,
        "ug_h2o_lambda": np.nan,
        "ug_h2o_beta": np.nan,
        "probe_layers": ",".join(str(index) for index in probe_layers),
        "avg_entropy_gate": np.nan,
    }


@torch.no_grad()
def free_run_uckv2_fixed_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: PromptExample,
    budget: int,
    max_new_tokens: int,
    config: UCKV2Config,
    sink: int = 4,
) -> Dict[str, Any]:
    encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    cuda_baseline_bytes = 0
    if device.type == "cuda":
        cuda_baseline_bytes = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)

    can_set_attention = hasattr(model, "set_attn_implementation")
    original_implementation = str(getattr(model.config, "_attn_implementation", ""))
    if can_set_attention:
        model.set_attn_implementation("eager")
    try:
        synchronize_device(device)
        prefill_start = time.perf_counter()
        past, current_logits, prefill_attentions = forward_prefill_with_attentions(
            model,
            input_ids,
        )
        prompt_len = int(input_ids.shape[1])
        salience = prefill_salience_from_attentions(
            prefill_attentions,
            prompt_len=prompt_len,
            query_tokens=config.prefill_query_tokens,
            probe_layers=config.probe_layers,
            device=device,
        )
        synchronize_device(device)
        prefill_elapsed_s = time.perf_counter() - prefill_start

        positions = list(range(prompt_len))
        scores = torch.zeros(prompt_len, dtype=torch.float32, device=device)
        generated: List[int] = []
        kept_counts: List[int] = []
        budget_overruns: List[int] = []
        cache_bytes: List[int] = []
        gate_values: List[float] = []
        peak_cache_bytes = cache_nbytes(past)
        attention_score_elapsed_s = 0.0
        utility_elapsed_s = 0.0
        topk_elapsed_s = 0.0
        slice_elapsed_s = 0.0
        eviction_count = 0
        decode_start = time.perf_counter()

        for step in range(max_new_tokens):
            abs_pos = prompt_len + step
            next_token = int(torch.argmax(current_logits, dim=-1).item())
            _, norm_entropy, _ = tensor_entropy_and_margin(current_logits)
            gate_value = entropy_gate(
                norm_entropy,
                config.entropy_gate_low,
                config.entropy_gate_high,
            )
            gate_values.append(gate_value)
            step_ids = torch.tensor([[next_token]], dtype=torch.long, device=device)
            attention_mask = torch.ones(
                (1, len(positions) + 1), dtype=torch.long, device=device
            )
            position_ids = torch.tensor([[abs_pos]], dtype=torch.long, device=device)
            output = model(
                input_ids=step_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
                **logits_limit_kwargs(model),
            )
            if output.attentions is None:
                raise RuntimeError("Eager attention did not return attention weights.")

            attention_start = time.perf_counter()
            step_scores = aggregate_attention_scores(
                output.attentions,
                probe_layers=config.probe_layers,
            ).to(scores.device)
            attention_score_elapsed_s += time.perf_counter() - attention_start
            if len(step_scores) != len(positions) + 1:
                raise RuntimeError(
                    "Attention/cache length mismatch in UCKV-2 fixed policy: "
                    f"attention={len(step_scores)} cache={len(positions) + 1}."
                )

            utility_start = time.perf_counter()
            gate_weight = 1.0 + config.lambda_gate * gate_value
            scores = torch.cat(
                [scores + gate_weight * step_scores[:-1], gate_weight * step_scores[-1:]]
            )
            salience = torch.cat(
                [salience, torch.zeros(1, dtype=torch.float32, device=salience.device)]
            )
            utility_elapsed_s += time.perf_counter() - utility_start

            past = output.past_key_values
            current_logits = output.logits[:, -1, :]
            positions.append(abs_pos)
            generated.append(next_token)
            peak_cache_bytes = max(peak_cache_bytes, cache_nbytes(past))

            should_evict = step == 0 or (step + 1) % config.evict_every == 0
            if should_evict:
                utility = scores + config.beta_salience * salience
                topk_start = time.perf_counter()
                keep = select_uckv2_heavy_hitter_indices(
                    utility,
                    positions,
                    sink=sink,
                    budget=budget,
                    current_abs_pos=abs_pos,
                    min_recent=config.min_recent,
                    recent_fraction=config.recent_fraction,
                )
                topk_elapsed_s += time.perf_counter() - topk_start

                slice_start = time.perf_counter()
                past = slice_past(past, keep)
                slice_elapsed_s += time.perf_counter() - slice_start
                keep_tensor = torch.tensor(keep, dtype=torch.long, device=scores.device)
                scores = scores.index_select(0, keep_tensor)
                salience = salience.index_select(0, keep_tensor)
                positions = [positions[index] for index in keep]
                eviction_count += 1

            kept_counts.append(len(positions))
            budget_overruns.append(max(0, len(positions) - budget))
            cache_bytes.append(cache_nbytes(past))

        synchronize_device(device)
        decode_elapsed_s = time.perf_counter() - decode_start
    finally:
        if can_set_attention:
            model.set_attn_implementation(original_implementation)

    elapsed_s = prefill_elapsed_s + decode_elapsed_s
    cuda_peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    selector_total_elapsed_s = (
        attention_score_elapsed_s
        + utility_elapsed_s
        + topk_elapsed_s
        + slice_elapsed_s
    )
    return {
        "prompt_id": example.prompt_id,
        "split": example.split,
        "task_type": example.task_type,
        "answer_position": example.answer_position,
        "context_words": example.context_words,
        "policy": f"uckv2_fixed_{budget}",
        "expected_answer": example.answer,
        "answer_contains": int(answer_in_text(example.answer, generated_text)),
        "generated_tokens": len(generated),
        "generated_text": generated_text,
        "elapsed_s": elapsed_s,
        "prefill_elapsed_s": prefill_elapsed_s,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tokens_per_s": (
            len(generated) / decode_elapsed_s if decode_elapsed_s > 0 else np.nan
        ),
        "peak_cache_bytes": peak_cache_bytes,
        "avg_cache_bytes": float(np.mean(cache_bytes)) if cache_bytes else np.nan,
        "cuda_peak_allocated_bytes": cuda_peak_bytes,
        "cuda_peak_delta_bytes": max(0, cuda_peak_bytes - cuda_baseline_bytes),
        "avg_kept_tokens": float(np.mean(kept_counts)) if kept_counts else np.nan,
        "max_kept_tokens": max(kept_counts) if kept_counts else np.nan,
        "budget_overrun_steps": sum(value > 0 for value in budget_overruns),
        "max_budget_overrun": max(budget_overruns) if budget_overruns else 0,
        "avg_selected_window": float(budget),
        "avg_selected_risk": np.nan,
        "fallback_steps": 0,
        "abstain_steps": 0,
        "selector_elapsed_s": selector_total_elapsed_s,
        "attention_score_elapsed_s": attention_score_elapsed_s,
        "utility_elapsed_s": utility_elapsed_s,
        "topk_elapsed_s": topk_elapsed_s,
        "slice_elapsed_s": slice_elapsed_s,
        "selector_decode_fraction": (
            selector_total_elapsed_s / decode_elapsed_s if decode_elapsed_s > 0 else np.nan
        ),
        "eviction_count": eviction_count,
        "evict_every": config.evict_every,
        "ug_h2o_lambda": config.lambda_gate,
        "ug_h2o_beta": config.beta_salience,
        "probe_layers": ",".join(str(index) for index in config.probe_layers),
        "avg_entropy_gate": float(np.mean(gate_values)) if gate_values else np.nan,
    }


@torch.no_grad()
def free_run_generation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    example: PromptExample,
    policy_name: str,
    max_new_tokens: int,
    risk_bundle: RiskModelBundle,
    candidate_windows: Sequence[int],
    uckv2_config: UCKV2Config,
) -> Dict[str, Any]:
    h2o_match = re.fullmatch(r"h2o_hh_(\d+)", policy_name)
    if h2o_match:
        return free_run_h2o_generation(
            model,
            tokenizer,
            device,
            example,
            budget=int(h2o_match.group(1)),
            max_new_tokens=max_new_tokens,
            sink=4,
        )
    h2o_matched = re.fullmatch(r"h2o_matched_(\d+)", policy_name)
    if h2o_matched:
        return free_run_h2o_generation(
            model,
            tokenizer,
            device,
            example,
            budget=int(h2o_matched.group(1)),
            max_new_tokens=max_new_tokens,
            sink=4,
            evict_every=uckv2_config.evict_every,
            recent_fraction=uckv2_config.recent_fraction,
            min_recent=uckv2_config.min_recent,
            probe_layers=uckv2_config.probe_layers,
            policy_label=policy_name,
        )
    uckv2_match = re.fullmatch(r"(?:uckv2_fixed|ugh2o_hh)_(\d+)", policy_name)
    if uckv2_match:
        return free_run_uckv2_fixed_generation(
            model,
            tokenizer,
            device,
            example,
            budget=int(uckv2_match.group(1)),
            max_new_tokens=max_new_tokens,
            config=uckv2_config,
            sink=4,
        )
    encoded = tokenizer(example.text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"].to(device)
    cuda_baseline_bytes = 0
    if device.type == "cuda":
        cuda_baseline_bytes = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
    synchronize_device(device)
    prefill_start = time.perf_counter()
    past, current_logits = forward_prefill(model, input_ids)
    synchronize_device(device)
    prefill_elapsed_s = time.perf_counter() - prefill_start
    prompt_len = int(input_ids.shape[1])
    positions = list(range(prompt_len))
    generated: List[int] = []
    selected_windows: List[int] = []
    selected_risks: List[float] = []
    selections: List[str] = []
    kept_counts: List[int] = []
    cache_bytes: List[int] = []
    decode_start = time.perf_counter()

    for step in range(max_new_tokens):
        current_abs_pos = prompt_len + step
        next_token = int(torch.argmax(current_logits, dim=-1).item())

        if policy_name == "full":
            keep_indices = list(range(len(positions)))
            selected_window = len(positions)
            selected_risk = float("nan")
            selection_reason = "full"
        elif policy_name in {"uckv_budget", "uckv_length_guard"}:
            if risk_bundle.status != "ok":
                raise RuntimeError("UCKV free-run requires a fitted risk model.")
            observed_length = prompt_len
            outside_calibration = not (
                risk_bundle.calibration_length_min - risk_bundle.calibration_length_margin
                <= observed_length
                <= risk_bundle.calibration_length_max + risk_bundle.calibration_length_margin
            )
            if policy_name == "uckv_length_guard" and outside_calibration:
                keep_indices = list(range(len(positions)))
                selected_window = len(positions)
                selected_risk = float("nan")
                selection_reason = "length_shift_abstain_full"
            else:
                keep_indices, selected_window, selected_risk, selection_reason = (
                    select_uckv_keep_indices(
                        positions,
                        current_abs_pos,
                        current_logits.detach().cpu(),
                        risk_bundle,
                        candidate_windows,
                        sink=4,
                    )
                )
        else:
            policy = parse_policy_name(policy_name)
            if policy.name.startswith("keynorm_hh_"):
                keep_indices = select_keynorm_heavy_hitter_indices(
                    past,
                    positions,
                    sink=policy.sink,
                    budget=policy.window,
                    current_abs_pos=current_abs_pos,
                )
            else:
                keep_indices = select_keep_indices(
                    positions,
                    sink=policy.sink,
                    window=policy.window,
                    current_abs_pos=current_abs_pos,
                )
            selected_window = policy.window
            selected_risk = float("nan")
            selection_reason = "fixed"

        past = slice_past(past, keep_indices)
        positions = [positions[i] for i in keep_indices]
        kept_counts.append(len(positions))
        past, current_logits = forward_one(model, next_token, past, positions, current_abs_pos, device)
        positions.append(current_abs_pos)
        cache_bytes.append(cache_nbytes(past))
        generated.append(next_token)
        selected_windows.append(selected_window)
        selected_risks.append(selected_risk)
        selections.append(selection_reason)

    synchronize_device(device)
    decode_elapsed_s = time.perf_counter() - decode_start
    elapsed_s = prefill_elapsed_s + decode_elapsed_s
    cuda_peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt_id": example.prompt_id,
        "split": example.split,
        "task_type": example.task_type,
        "answer_position": example.answer_position,
        "context_words": example.context_words,
        "policy": policy_name,
        "expected_answer": example.answer,
        "answer_contains": int(answer_in_text(example.answer, generated_text)),
        "generated_tokens": len(generated),
        "generated_text": generated_text,
        "elapsed_s": elapsed_s,
        "prefill_elapsed_s": prefill_elapsed_s,
        "decode_elapsed_s": decode_elapsed_s,
        "decode_tokens_per_s": (
            len(generated) / decode_elapsed_s if decode_elapsed_s > 0 else np.nan
        ),
        "peak_cache_bytes": max(cache_bytes) if cache_bytes else 0,
        "avg_cache_bytes": float(np.mean(cache_bytes)) if cache_bytes else np.nan,
        "cuda_peak_allocated_bytes": cuda_peak_bytes,
        "cuda_peak_delta_bytes": max(0, cuda_peak_bytes - cuda_baseline_bytes),
        "avg_kept_tokens": float(np.mean(kept_counts)) if kept_counts else np.nan,
        "max_kept_tokens": max(kept_counts) if kept_counts else np.nan,
        "budget_overrun_steps": np.nan,
        "max_budget_overrun": np.nan,
        "avg_selected_window": float(np.mean(selected_windows)) if selected_windows else np.nan,
        "avg_selected_risk": safe_nanmean(selected_risks),
        "fallback_steps": sum(1 for value in selections if value == "fallback_max_budget"),
        "abstain_steps": sum(
            1 for value in selections if value == "length_shift_abstain_full"
        ),
        "selector_elapsed_s": np.nan,
        "attention_score_elapsed_s": np.nan,
        "utility_elapsed_s": np.nan,
        "topk_elapsed_s": np.nan,
        "slice_elapsed_s": np.nan,
        "selector_decode_fraction": np.nan,
        "eviction_count": np.nan,
        "evict_every": np.nan,
        "ug_h2o_lambda": np.nan,
        "ug_h2o_beta": np.nan,
        "probe_layers": "",
        "avg_entropy_gate": np.nan,
    }


def summarize_steps(steps: pd.DataFrame) -> pd.DataFrame:
    grouped = steps.groupby(["split", "policy"], as_index=False)
    return grouped.agg(
        prompts=("prompt_id", "nunique"),
        steps=("step", "count"),
        avg_kl=("kl_full_to_compressed", "mean"),
        p95_kl=("kl_full_to_compressed", lambda s: float(np.percentile(s, 95))),
        max_kl=("kl_full_to_compressed", "max"),
        top1_mismatch_rate=("top1_mismatch", "mean"),
        avg_retained_ratio=("retained_ratio", "mean"),
        avg_kept_tokens=("kept_tokens_before", "mean"),
        avg_full_tokens=("full_tokens_before", "mean"),
        avg_replay_s_per_step=("replay_elapsed_s_per_step", "mean"),
    )


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    uckv_candidate_windows = parse_windows(args.uckv_candidate_windows)
    keynorm_heavy_hitter_budgets = (
        parse_windows(args.keynorm_heavy_hitter_budgets)
        if args.keynorm_heavy_hitter_budgets.strip()
        else []
    )
    h2o_heavy_hitter_budgets = (
        parse_windows(args.h2o_heavy_hitter_budgets)
        if args.h2o_heavy_hitter_budgets.strip()
        else []
    )
    if args.uckv2_evict_every <= 0:
        raise ValueError("--uckv2-evict-every must be positive.")
    if args.uckv2_min_recent <= 0:
        raise ValueError("--uckv2-min-recent must be positive.")
    if args.uckv2_prefill_query_tokens <= 0:
        raise ValueError("--uckv2-prefill-query-tokens must be positive.")
    if args.uckv2_recent_fraction <= 0:
        raise ValueError("--uckv2-recent-fraction must be positive.")
    uckv2_probe_layers = parse_optional_indices(args.uckv2_probe_layers)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = args.model.replace("/", "__")
    label = slugify(args.run_label)
    dir_parts = [run_id]
    if label:
        dir_parts.append(label)
    dir_parts.append(model_slug)
    output_dir = Path(args.output_root) / "_".join(dir_parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "model": args.model,
        "prompts": args.prompts,
        "run_label": args.run_label,
        "task_filter": parse_csv_names(args.task_filter),
        "answer_position_filter": parse_csv_names(args.answer_position_filter),
        "max_new_tokens": args.max_new_tokens,
        "max_prompts": args.max_prompts,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "apply_chat_template": args.apply_chat_template,
        "kl_risk_threshold": args.kl_risk_threshold,
        "uckv_risk_tolerance": args.uckv_risk_tolerance,
        "uckv_candidate_windows": uckv_candidate_windows,
        "keynorm_heavy_hitter_budgets": keynorm_heavy_hitter_budgets,
        "h2o_heavy_hitter_budgets": h2o_heavy_hitter_budgets,
        "uckv2_probe_layers": list(uckv2_probe_layers),
        "uckv2_lambda": args.uckv2_lambda,
        "uckv2_beta": args.uckv2_beta,
        "uckv2_evict_every": args.uckv2_evict_every,
        "uckv2_recent_fraction": args.uckv2_recent_fraction,
        "uckv2_min_recent": args.uckv2_min_recent,
        "uckv2_prefill_query_tokens": args.uckv2_prefill_query_tokens,
        "enable_length_guard": args.enable_length_guard,
        "free_run_eval": args.free_run_eval,
        "free_run_policies": parse_csv_names(args.free_run_policies),
        "progress_every": args.progress_every,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "risk_model_backend": "standardized_numpy_newton_logistic",
    }
    write_json(output_dir / "config.json", config)

    prompts = load_prompts(
        Path(args.prompts),
        args.max_prompts,
        args.split_mode,
        parse_csv_names(args.task_filter),
        parse_csv_names(args.answer_position_filter),
    )
    if not prompts:
        raise ValueError("No prompts matched the requested prompt file, limit, and filters.")
    print(f"Loaded {len(prompts)} prompts. Output directory: {output_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.apply_chat_template:
        prompts = apply_chat_template_to_prompts(prompts, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    traces: Dict[str, FullTrace] = {}
    trace_rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(prompts, start=1):
        trace = build_full_trace(model, tokenizer, device, example, args.max_new_tokens)
        traces[example.prompt_id] = trace
        trace_rows.append(
            {
                "prompt_id": example.prompt_id,
                "split": example.split,
                "prompt_len": trace.prompt_len,
                "task_type": example.task_type,
                "answer_position": example.answer_position,
                "context_words": example.context_words,
                "expected_answer": example.answer,
                "full_answer_contains": int(answer_in_text(example.answer, trace.generated_text)),
                "generated_tokens": len(trace.generated_ids),
                "full_elapsed_s": trace.elapsed_s,
                "full_prefill_elapsed_s": trace.prefill_elapsed_s,
                "full_decode_elapsed_s": trace.decode_elapsed_s,
                "full_decode_tokens_per_s": (
                    len(trace.generated_ids) / trace.decode_elapsed_s
                    if trace.decode_elapsed_s > 0
                    else np.nan
                ),
                "full_peak_cache_bytes": trace.peak_cache_bytes,
                "generated_text": trace.generated_text,
            }
        )
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(prompts)):
            print(f"[full-trace] {idx}/{len(prompts)} prompts", flush=True)

    calibration_entropies: List[float] = []
    for example in prompts:
        if example.split != "calibration":
            continue
        for logits in traces[example.prompt_id].decision_logits:
            _, norm_entropy, _ = tensor_entropy_and_margin(logits)
            calibration_entropies.append(norm_entropy)
    entropy_low_q = float(np.quantile(calibration_entropies, 0.33))
    entropy_high_q = float(np.quantile(calibration_entropies, 0.66))
    entropy_gate_low_q = float(np.quantile(calibration_entropies, 0.50))
    entropy_gate_high_q = float(np.quantile(calibration_entropies, 0.90))
    uckv2_config = UCKV2Config(
        lambda_gate=args.uckv2_lambda,
        beta_salience=args.uckv2_beta,
        evict_every=args.uckv2_evict_every,
        recent_fraction=args.uckv2_recent_fraction,
        min_recent=args.uckv2_min_recent,
        prefill_query_tokens=args.uckv2_prefill_query_tokens,
        probe_layers=uckv2_probe_layers,
        entropy_gate_low=entropy_gate_low_q,
        entropy_gate_high=entropy_gate_high_q,
    )

    training_policy_names = [sink_window_policy_name(window) for window in uckv_candidate_windows]
    policies = [
        Policy("full", sink=0, window=0),
        Policy("window_16", sink=0, window=16),
        *[
            Policy(sink_window_policy_name(window), sink=4, window=window)
            for window in uckv_candidate_windows
        ],
        *[
            Policy(f"keynorm_hh_{budget}", sink=4, window=budget)
            for budget in keynorm_heavy_hitter_budgets
        ],
        Policy("uckv_entropy_adaptive", sink=4, window=16, adaptive=True),
    ]

    step_rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(prompts, start=1):
        for policy in policies:
            step_rows.extend(
                replay_policy(
                    model,
                    tokenizer,
                    device,
                    example,
                    traces[example.prompt_id],
                    policy,
                    entropy_low_q=entropy_low_q,
                    entropy_high_q=entropy_high_q,
                )
            )
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(prompts)):
            print(f"[policy-replay] {idx}/{len(prompts)} prompts", flush=True)

    steps = pd.DataFrame(step_rows)
    risk_predictions, risk_summary, risk_bundle = fit_risk_model(
        steps,
        args.kl_risk_threshold,
        args.uckv_risk_tolerance,
        training_policy_names,
    )
    if risk_bundle.status == "ok":
        for idx, example in enumerate(prompts, start=1):
            step_rows.extend(
                replay_uckv_budget(
                    model,
                    tokenizer,
                    device,
                    example,
                    traces[example.prompt_id],
                    risk_bundle,
                    candidate_windows=uckv_candidate_windows,
                    sink=4,
                )
            )
            if args.enable_length_guard:
                step_rows.extend(
                    replay_uckv_budget(
                        model,
                        tokenizer,
                        device,
                        example,
                        traces[example.prompt_id],
                        risk_bundle,
                        candidate_windows=uckv_candidate_windows,
                        sink=4,
                        length_guard=True,
                    )
                )
            if args.progress_every > 0 and (
                idx % args.progress_every == 0 or idx == len(prompts)
            ):
                print(f"[uckv-replay] {idx}/{len(prompts)} prompts", flush=True)
        steps = pd.DataFrame(step_rows)

    summary = summarize_steps(steps)
    if "uckv_selection" in steps.columns and steps["policy"].eq("uckv_budget").any():
        uckv_selection_summary = (
            steps[steps["policy"].eq("uckv_budget")]
            .groupby(["split", "window", "uckv_selection"], as_index=False)
            .agg(
                steps=("step", "count"),
                avg_pred_risk=("uckv_pred_risk", "mean"),
                avg_retained_ratio=("retained_ratio", "mean"),
                top1_mismatch_rate=("top1_mismatch", "mean"),
                avg_kl=("kl_full_to_compressed", "mean"),
            )
        )
    else:
        uckv_selection_summary = pd.DataFrame(
            [{"status": risk_bundle.status, "reason": risk_bundle.reason}]
        )
    trace_df = pd.DataFrame(trace_rows)
    free_run_df = pd.DataFrame()
    if args.free_run_eval:
        free_run_rows: List[Dict[str, Any]] = []
        for idx, example in enumerate(prompts, start=1):
            for policy_name in parse_csv_names(args.free_run_policies):
                if policy_name in {"uckv_budget", "uckv_length_guard"} and (
                    risk_bundle.status != "ok"
                ):
                    continue
                free_run_rows.append(
                    free_run_generation(
                        model,
                        tokenizer,
                        device,
                        example,
                        policy_name,
                        args.max_new_tokens,
                        risk_bundle,
                        uckv_candidate_windows,
                        uckv2_config,
                    )
                )
            if args.progress_every > 0 and (
                idx % args.progress_every == 0 or idx == len(prompts)
            ):
                print(f"[free-run] {idx}/{len(prompts)} prompts", flush=True)
        free_run_df = pd.DataFrame(free_run_rows)

    trace_df.to_csv(output_dir / "full_traces.csv", index=False)
    steps.to_csv(output_dir / "steps.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    risk_predictions.to_csv(output_dir / "risk_predictions.csv", index=False)
    risk_summary.to_csv(output_dir / "risk_summary.csv", index=False)
    uckv_selection_summary.to_csv(output_dir / "uckv_selection_summary.csv", index=False)
    if args.free_run_eval:
        free_run_df.to_csv(output_dir / "free_run_generations.csv", index=False)
        if len(free_run_df):
            free_run_summary = (
                free_run_df.groupby(["split", "policy", "task_type"], as_index=False)
                .agg(
                    prompts=("prompt_id", "count"),
                    answer_contains=("answer_contains", "mean"),
                    avg_kept_tokens=("avg_kept_tokens", "mean"),
                    max_kept_tokens=("max_kept_tokens", "max"),
                    budget_overrun_steps=("budget_overrun_steps", "sum"),
                    max_budget_overrun=("max_budget_overrun", "max"),
                    fallback_steps=("fallback_steps", "sum"),
                    abstain_steps=("abstain_steps", "sum"),
                    avg_prefill_s=("prefill_elapsed_s", "mean"),
                    avg_decode_s=("decode_elapsed_s", "mean"),
                    decode_tokens_per_s=("decode_tokens_per_s", "mean"),
                    peak_cache_bytes=("peak_cache_bytes", "max"),
                    avg_cache_bytes=("avg_cache_bytes", "mean"),
                    cuda_peak_delta_bytes=("cuda_peak_delta_bytes", "max"),
                    avg_selector_elapsed_s=("selector_elapsed_s", "mean"),
                    avg_selector_decode_fraction=("selector_decode_fraction", "mean"),
                    avg_eviction_count=("eviction_count", "mean"),
                    avg_entropy_gate=("avg_entropy_gate", "mean"),
                )
            )
        else:
            free_run_summary = pd.DataFrame()
        free_run_summary.to_csv(output_dir / "free_run_summary.csv", index=False)
    write_json(
        output_dir / "adaptive_thresholds.json",
        {
            "norm_entropy_low_q33": entropy_low_q,
            "norm_entropy_high_q66": entropy_high_q,
            "uckv2_entropy_gate_low_q50": entropy_gate_low_q,
            "uckv2_entropy_gate_high_q90": entropy_gate_high_q,
            "source": "calibration split decision logits",
        },
    )

    print(f"Output directory: {output_dir}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nRisk model:")
    if len(risk_summary):
        print(risk_summary.to_string(index=False))
    else:
        print("No risk summary rows produced.")
    print("\nUCKV selection:")
    if len(uckv_selection_summary):
        print(uckv_selection_summary.to_string(index=False))
    else:
        print("No UCKV selection summary rows produced.")
    if args.free_run_eval:
        print("\nFree-run task eval:")
        if len(free_run_df):
            print(free_run_summary.to_string(index=False))
        else:
            print("No free-run rows produced.")


if __name__ == "__main__":
    main()
