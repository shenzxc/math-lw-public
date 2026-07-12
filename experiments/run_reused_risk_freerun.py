#!/usr/bin/env python3
"""Run UCKV free-run sweeps by reusing fixed-window replay from a prior run.

This is for fine-grained tolerance tuning after a full replay run already
exists. It refits the logistic risk model from the source run's `steps.csv`,
then runs only UCKV free-generation for each requested tolerance.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_kv_cache_pilot import (
    apply_chat_template_to_prompts,
    fit_risk_model,
    free_run_generation,
    load_prompts,
    parse_csv_names,
    select_device,
    select_dtype,
    set_seed,
    sink_window_policy_name,
    slugify,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-root", default="outputs/experiments")
    parser.add_argument("--run-label-prefix", default="")
    parser.add_argument("--uckv-risk-tolerances", required=True)
    parser.add_argument("--free-run-policies", default="uckv_budget")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=12)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_tolerances(value: str) -> List[float]:
    tolerances = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not tolerances or any(tolerance < 0 or tolerance > 1 for tolerance in tolerances):
        raise ValueError("--uckv-risk-tolerances must contain values in [0, 1].")
    return tolerances


def tolerance_slug(tolerance: float) -> str:
    return f"tol{str(tolerance).replace('.', '')}"


def write_free_run_summary(free_run_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    free_run_df.to_csv(output_dir / "free_run_generations.csv", index=False)
    if len(free_run_df):
        summary = (
            free_run_df.groupby(["split", "policy", "task_type"], as_index=False)
            .agg(
                prompts=("prompt_id", "count"),
                answer_contains=("answer_contains", "mean"),
                avg_kept_tokens=("avg_kept_tokens", "mean"),
                fallback_steps=("fallback_steps", "sum"),
                abstain_steps=("abstain_steps", "sum"),
                avg_prefill_s=("prefill_elapsed_s", "mean"),
                avg_decode_s=("decode_elapsed_s", "mean"),
                decode_tokens_per_s=("decode_tokens_per_s", "mean"),
                peak_cache_bytes=("peak_cache_bytes", "max"),
                avg_cache_bytes=("avg_cache_bytes", "mean"),
                cuda_peak_delta_bytes=("cuda_peak_delta_bytes", "max"),
            )
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(output_dir / "free_run_summary.csv", index=False)
    return summary


def fixed_window(policy: str) -> int | None:
    prefix = "sink4_window_"
    if not str(policy).startswith(prefix):
        return None
    return int(str(policy)[len(prefix) :])


def simulate_uckv_from_fixed_replay(
    steps: pd.DataFrame,
    risk_predictions: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_steps = steps[steps["policy"].str.startswith("sink4_window_")].copy()
    fixed_steps["window"] = fixed_steps["policy"].map(fixed_window)
    risks = risk_predictions.copy()
    risks["window"] = risks["policy"].map(fixed_window)
    fixed_steps = fixed_steps[fixed_steps["window"].notna()].copy()
    risks = risks[risks["window"].notna()].copy()
    merged = fixed_steps.merge(
        risks[["prompt_id", "split", "policy", "step", "risk_pred"]],
        on=["prompt_id", "split", "policy", "step"],
        how="inner",
    )
    selected_rows: List[pd.Series] = []
    for _, group in merged.sort_values(["prompt_id", "split", "step", "window"]).groupby(
        ["prompt_id", "split", "step"],
        sort=False,
    ):
        safe = group[group["risk_pred"] <= tolerance]
        if len(safe):
            selected = safe.sort_values(["kept_tokens_before", "window"]).iloc[0].copy()
            selected["uckv_selection"] = "min_safe_budget"
        else:
            selected = group.sort_values(["kept_tokens_before", "window"]).iloc[-1].copy()
            selected["uckv_selection"] = "fallback_max_budget"
        selected["policy"] = "uckv_budget"
        selected["uckv_pred_risk"] = selected["risk_pred"]
        selected["uckv_risk_tolerance"] = tolerance
        selected_rows.append(selected)

    simulated_steps = pd.DataFrame(selected_rows)
    summary = (
        simulated_steps.groupby(["split", "policy"], as_index=False)
        .agg(
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
            uckv_fallback_steps=(
                "uckv_selection",
                lambda s: int((s == "fallback_max_budget").sum()),
            ),
        )
    )
    return simulated_steps, summary


def make_output_dir(output_root: Path, run_label: str, model: str) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_slug = model.replace("/", "__")
    label = slugify(run_label)
    parts = [run_id]
    if label:
        parts.append(label)
    parts.append(model_slug)
    output_dir = output_root / "_".join(parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    source_run = Path(args.source_run)
    source_config = read_json(source_run / "config.json")
    source_steps = pd.read_csv(source_run / "steps.csv")
    set_seed(int(source_config.get("seed", 7)))

    model_name = str(source_config["model"])
    candidate_windows = [int(window) for window in source_config["uckv_candidate_windows"]]
    training_policy_names = [sink_window_policy_name(window) for window in candidate_windows]
    prompts = load_prompts(
        Path(source_config["prompts"]),
        int(source_config["max_prompts"]),
        str(source_config["split_mode"]),
        source_config.get("task_filter", []),
        source_config.get("answer_position_filter", []),
    )

    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if source_config.get("apply_chat_template"):
        prompts = apply_chat_template_to_prompts(prompts, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()

    output_root = Path(args.output_root)
    free_run_policies = parse_csv_names(args.free_run_policies)
    for tolerance in parse_tolerances(args.uckv_risk_tolerances):
        risk_predictions, risk_summary, risk_bundle = fit_risk_model(
            source_steps,
            float(source_config["kl_risk_threshold"]),
            tolerance,
            training_policy_names,
        )
        if risk_bundle.status != "ok":
            raise RuntimeError(f"Risk model unavailable at tolerance {tolerance}: {risk_bundle.reason}")

        prefix = args.run_label_prefix or f"{source_config.get('run_label', '')}_reuse"
        run_label = f"{prefix}_{tolerance_slug(tolerance)}"
        output_dir = make_output_dir(output_root, run_label, model_name)
        config = {
            **source_config,
            "run_label": run_label,
            "source_run": str(source_run),
            "reused_risk_from": str(source_run / "steps.csv"),
            "replay_metrics_source": "fixed_candidate_simulation",
            "uckv_risk_tolerance": tolerance,
            "free_run_policies": free_run_policies,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        }
        write_json(output_dir / "config.json", config)
        risk_predictions.to_csv(output_dir / "risk_predictions.csv", index=False)
        risk_summary.to_csv(output_dir / "risk_summary.csv", index=False)
        simulated_steps, simulated_summary = simulate_uckv_from_fixed_replay(
            source_steps,
            risk_predictions,
            tolerance,
        )
        simulated_steps.to_csv(output_dir / "simulated_steps.csv", index=False)
        simulated_summary.to_csv(output_dir / "simulated_summary.csv", index=False)

        print(
            f"Running {run_label} with {len(prompts)} prompts. Output directory: {output_dir}",
            flush=True,
        )
        free_run_rows: List[Dict[str, Any]] = []
        for idx, example in enumerate(prompts, start=1):
            for policy_name in free_run_policies:
                free_run_rows.append(
                    free_run_generation(
                        model,
                        tokenizer,
                        device,
                        example,
                        policy_name,
                        int(source_config["max_new_tokens"]),
                        risk_bundle,
                        candidate_windows,
                    )
                )
            if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(prompts)):
                print(f"[{run_label} free-run] {idx}/{len(prompts)} prompts", flush=True)

        free_run_df = pd.DataFrame(free_run_rows)
        free_run_summary = write_free_run_summary(free_run_df, output_dir)
        print("\nFree-run task eval:")
        print(free_run_summary.to_string(index=False) if len(free_run_summary) else "No rows.")
        print("\nSimulated replay summary:")
        print(simulated_summary.to_string(index=False))


if __name__ == "__main__":
    main()
