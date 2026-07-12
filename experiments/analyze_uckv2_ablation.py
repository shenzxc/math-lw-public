#!/usr/bin/env python3
"""Aggregate UCKV2 ablation runs by configuration tag."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--outdir", default="outputs/analysis_uckv2_ablation")
    parser.add_argument("--label-contains", default="uckv2_ablate_")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ablation_tag(run_label: str) -> str:
    match = re.search(r"uckv2_ablate_(.+?)_seed\d+", run_label)
    return match.group(1) if match else run_label or "unknown"


def policy_budget(policy: str) -> int | None:
    match = re.search(r"_(\d+)$", policy)
    return int(match.group(1)) if match else None


def collect_run_dirs(root: Path, label_contains: str) -> list[Path]:
    run_dirs: list[Path] = []
    for config_path in sorted(root.glob("*/config.json")):
        config = read_json(config_path)
        run_label = str(config.get("run_label", ""))
        if label_contains and label_contains not in run_label:
            continue
        if not (config_path.parent / "free_run_summary.csv").exists():
            continue
        run_dirs.append(config_path.parent)
    return run_dirs


def standard_error(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def add_metadata(frame: pd.DataFrame, run_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    run_label = str(config.get("run_label", ""))
    out.insert(0, "run_dir", str(run_dir))
    out.insert(1, "run_label", run_label)
    out.insert(2, "ablation", ablation_tag(run_label))
    out.insert(3, "model", config.get("model", ""))
    out.insert(4, "max_prompts", config.get("max_prompts"))
    out.insert(5, "max_new_tokens", config.get("max_new_tokens"))
    out.insert(6, "uckv2_lambda", config.get("uckv2_lambda"))
    out.insert(7, "uckv2_beta", config.get("uckv2_beta"))
    out.insert(8, "uckv2_evict_every", config.get("uckv2_evict_every"))
    out.insert(9, "uckv2_min_recent", config.get("uckv2_min_recent"))
    out.insert(10, "uckv2_recent_fraction", config.get("uckv2_recent_fraction"))
    return out


def collect_summaries(run_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json")
        frame = pd.read_csv(run_dir / "free_run_summary.csv")
        frame = add_metadata(frame, run_dir, config)
        frame["budget"] = frame["policy"].map(policy_budget)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate(summaries: pd.DataFrame) -> pd.DataFrame:
    eval_rows = summaries[summaries["split"].eq("evaluation")].copy()
    if eval_rows.empty:
        return pd.DataFrame()
    return (
        eval_rows.groupby(
            [
                "ablation",
                "policy",
                "budget",
                "uckv2_lambda",
                "uckv2_beta",
                "uckv2_evict_every",
                "uckv2_min_recent",
                "uckv2_recent_fraction",
            ],
            dropna=False,
        )
        .agg(
            runs=("run_label", "nunique"),
            answer_contains=("answer_contains", "mean"),
            answer_contains_se=("answer_contains", standard_error),
            avg_kept_tokens=("avg_kept_tokens", "mean"),
            decode_tokens_per_s=("decode_tokens_per_s", "mean"),
            selector_decode_fraction=("avg_selector_decode_fraction", "mean"),
        )
        .reset_index()
        .sort_values(["ablation", "budget", "policy"])
    )


def matched_deltas(policy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    columns = [
        "ablation",
        "budget",
        "h2o_answer_contains",
        "uckv2_answer_contains",
        "answer_delta",
        "h2o_avg_kept_tokens",
        "uckv2_avg_kept_tokens",
        "kept_delta",
        "uckv2_lambda",
        "uckv2_beta",
        "uckv2_evict_every",
        "uckv2_selector_decode_fraction",
    ]
    if policy_summary.empty:
        return pd.DataFrame(columns=columns)
    for ablation, group in policy_summary.groupby("ablation"):
        for budget in sorted(group["budget"].dropna().unique()):
            h2o = group[group["policy"].eq(f"h2o_hh_{int(budget)}")]
            uckv2 = group[group["policy"].eq(f"uckv2_fixed_{int(budget)}")]
            if h2o.empty or uckv2.empty:
                continue
            h = h2o.iloc[0]
            u = uckv2.iloc[0]
            rows.append(
                {
                    "ablation": ablation,
                    "budget": int(budget),
                    "h2o_answer_contains": h["answer_contains"],
                    "uckv2_answer_contains": u["answer_contains"],
                    "answer_delta": u["answer_contains"] - h["answer_contains"],
                    "h2o_avg_kept_tokens": h["avg_kept_tokens"],
                    "uckv2_avg_kept_tokens": u["avg_kept_tokens"],
                    "kept_delta": u["avg_kept_tokens"] - h["avg_kept_tokens"],
                    "uckv2_lambda": u["uckv2_lambda"],
                    "uckv2_beta": u["uckv2_beta"],
                    "uckv2_evict_every": u["uckv2_evict_every"],
                    "uckv2_selector_decode_fraction": u["selector_decode_fraction"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["budget", "answer_delta"], ascending=[True, False]
    )


def write_markdown(policy_summary: pd.DataFrame, deltas: pd.DataFrame, outpath: Path) -> None:
    if policy_summary.empty:
        outpath.write_text("No completed UCKV2 ablation runs found.\n", encoding="utf-8")
        return

    best_by_budget = (
        deltas.sort_values(["budget", "answer_delta"], ascending=[True, False])
        .groupby("budget", as_index=False)
        .head(1)
    )
    text = [
        "# UCKV2 Ablation Results",
        "",
        "Evaluation split only. Rows are grouped by ablation tag and matched budget.",
        "",
        "## Best Ablation By Budget",
        "",
        best_by_budget.to_markdown(index=False, floatfmt=".4f")
        if not best_by_budget.empty
        else "No matched H2O/UCKV2 budget rows found.",
        "",
        "## Matched-Budget Deltas",
        "",
        deltas.to_markdown(index=False, floatfmt=".4f") if not deltas.empty else "No deltas.",
        "",
        "## Policy Summary",
        "",
        policy_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    outpath.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_run_dirs(Path(args.root), args.label_contains)
    summaries = collect_summaries(run_dirs)
    policy_summary = aggregate(summaries)
    deltas = matched_deltas(policy_summary)

    summaries.to_csv(outdir / "seed_policy_summary.csv", index=False)
    policy_summary.to_csv(outdir / "policy_summary.csv", index=False)
    deltas.to_csv(outdir / "matched_budget_deltas.csv", index=False)
    write_markdown(policy_summary, deltas, outdir / "uckv2_ablation_results.md")

    print(f"Run dirs: {len(run_dirs)}")
    print(f"Wrote {outdir}")
    if not deltas.empty:
        print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
