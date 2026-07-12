#!/usr/bin/env python3
"""Analyze retrieval-only UCKV free-run and replay tradeoffs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--outdir", default="outputs/analysis_retrieval")
    parser.add_argument("--label-contains", default="retrieval")
    parser.add_argument("--tolerances", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    return parser.parse_args()


def parse_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_format(config: Dict[str, Any]) -> str:
    return "chat_template" if config.get("apply_chat_template") else "raw_prompt"


def run_metadata(run_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "run_label": config.get("run_label", ""),
        "model": config.get("model", ""),
        "prompt_format": prompt_format(config),
        "max_new_tokens": config.get("max_new_tokens"),
        "kl_risk_threshold": config.get("kl_risk_threshold"),
        "uckv_risk_tolerance": config.get("uckv_risk_tolerance"),
    }


def collect_run_dirs(root: Path, label_contains: str) -> List[Path]:
    run_dirs: List[Path] = []
    for config_path in sorted(root.glob("*/config.json")):
        config = read_json(config_path)
        run_label = str(config.get("run_label", ""))
        if label_contains and label_contains not in run_label:
            continue
        run_dirs.append(config_path.parent)
    return run_dirs


def prepend_metadata(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for key, value in reversed(list(metadata.items())):
        out.insert(0, key, value)
    return out


def collect_free_run(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "free_run_generations.csv"
        if not path.exists():
            continue
        config = read_json(run_dir / "config.json")
        frames.append(prepend_metadata(pd.read_csv(path), run_metadata(run_dir, config)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_replay_summary(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "summary.csv"
        metric_source = "actual_replay"
        if not path.exists():
            path = run_dir / "simulated_summary.csv"
            metric_source = "fixed_candidate_simulation"
        if not path.exists():
            continue
        config = read_json(run_dir / "config.json")
        frame = prepend_metadata(pd.read_csv(path), run_metadata(run_dir, config))
        frame["replay_metric_source"] = metric_source
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_free_run(free_run: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if free_run.empty:
        return pd.DataFrame()
    return (
        free_run.groupby(list(group_cols), as_index=False)
        .agg(
            prompts=("prompt_id", "count"),
            answer_contains=("answer_contains", "mean"),
            avg_kept_tokens=("avg_kept_tokens", "mean"),
            fallback_steps=("fallback_steps", "sum"),
        )
        .sort_values(list(group_cols))
    )


def make_tradeoff(free_by_policy: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    if free_by_policy.empty:
        return pd.DataFrame()
    eval_free = free_by_policy[free_by_policy["split"].eq("evaluation")].copy()
    if replay.empty:
        return eval_free
    replay_cols = [
        "run_dir",
        "policy",
        "split",
        "avg_kl",
        "top1_mismatch_rate",
        "avg_retained_ratio",
        "avg_full_tokens",
        "replay_metric_source",
    ]
    existing = [col for col in replay_cols if col in replay.columns]
    eval_replay = replay[replay["split"].eq("evaluation")][existing].copy()
    return eval_free.merge(eval_replay, on=["run_dir", "policy", "split"], how="left")


def parse_window(policy: str) -> int | None:
    match = re.fullmatch(r"sink4_window_(\d+)", str(policy))
    return int(match.group(1)) if match else None


def simulate_tolerance_sweep(run_dirs: Sequence[Path], tolerances: Sequence[float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        risk_path = run_dir / "risk_predictions.csv"
        steps_path = run_dir / "steps.csv"
        if not risk_path.exists() or not steps_path.exists():
            continue
        config = read_json(run_dir / "config.json")
        metadata = run_metadata(run_dir, config)
        risks = pd.read_csv(risk_path)
        steps = pd.read_csv(steps_path)
        risks["window"] = risks["policy"].map(parse_window)
        risks = risks[risks["window"].notna()].copy()
        steps["window"] = steps["policy"].map(parse_window)
        steps = steps[steps["window"].notna()].copy()
        merged = steps.merge(
            risks[["prompt_id", "split", "policy", "step", "risk_pred"]],
            on=["prompt_id", "split", "policy", "step"],
            how="inner",
        )
        if merged.empty:
            continue
        sort_cols = ["prompt_id", "split", "step", "window"]
        merged = merged.sort_values(sort_cols)
        group_cols = ["prompt_id", "split", "step"]
        for tolerance in tolerances:
            selected_rows: List[pd.Series] = []
            for _, group in merged.groupby(group_cols, sort=False):
                safe = group[group["risk_pred"] <= tolerance]
                if len(safe):
                    selected_rows.append(safe.sort_values("window").iloc[0])
                else:
                    selected_rows.append(group.sort_values("window").iloc[-1])
            selected = pd.DataFrame(selected_rows)
            selected["fallback"] = selected["risk_pred"] > tolerance
            for split, split_rows in selected.groupby("split"):
                rows.append(
                    {
                        **metadata,
                        "simulated_tolerance": tolerance,
                        "split": split,
                        "steps": len(split_rows),
                        "avg_kl": split_rows["kl_full_to_compressed"].mean(),
                        "top1_mismatch_rate": split_rows["top1_mismatch"].mean(),
                        "avg_retained_ratio": split_rows["retained_ratio"].mean(),
                        "avg_kept_tokens": split_rows["kept_tokens_before"].mean(),
                        "fallback_steps": int(split_rows["fallback"].sum()),
                        "avg_selected_window": split_rows["window"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def write_markdown(tradeoff: pd.DataFrame, outpath: Path) -> None:
    if tradeoff.empty:
        outpath.write_text("No retrieval free-run tradeoff rows found.\n", encoding="utf-8")
        return
    cols = [
        "run_label",
        "model",
        "prompt_format",
        "policy",
        "answer_contains",
        "avg_kept_tokens",
        "avg_kl",
        "top1_mismatch_rate",
        "avg_retained_ratio",
        "fallback_steps",
        "replay_metric_source",
    ]
    existing = [col for col in cols if col in tradeoff.columns]
    text = [
        "# Retrieval Free-Run Tradeoff",
        "",
        "Evaluation split only. Task score is exact answer containment in generated text.",
        "",
        tradeoff[existing].sort_values(["run_label", "policy"]).to_markdown(index=False),
        "",
    ]
    outpath.write_text("\n".join(text), encoding="utf-8")


def write_tradeoff_plot(tradeoff: pd.DataFrame, outpath: Path) -> None:
    if tradeoff.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    tradeoff = tradeoff.copy()
    tradeoff["panel"] = tradeoff["model"].str.split("/").str[-1] + "\n" + tradeoff["prompt_format"]
    panels = list(tradeoff["panel"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(5.2 * len(panels), 4.2),
        squeeze=False,
        sharey=True,
    )
    for ax, panel in zip(axes[0], panels):
        rows = tradeoff[tradeoff["panel"].eq(panel)].copy()
        for _, row in rows.iterrows():
            ax.scatter(row["avg_kept_tokens"], row["answer_contains"], s=42)
            label = str(row["policy"]).replace("sink4_window_", "w")
            if row["policy"] == "uckv_budget":
                label = f"uckv t={row['uckv_risk_tolerance']}"
            offset = (-34, -13) if label.startswith("uckv") else (4, 4)
            ax.annotate(
                label,
                (row["avg_kept_tokens"], row["answer_contains"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_title(panel)
        ax.set_xlabel("Avg kept tokens")
        ax.grid(True, alpha=0.25)
    axes[0][0].set_ylabel("Answer containment")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_dirs = collect_run_dirs(Path(args.root), args.label_contains)

    free_run = collect_free_run(run_dirs)
    replay = collect_replay_summary(run_dirs)
    metadata_cols = [
        "run_dir",
        "run_label",
        "model",
        "prompt_format",
        "max_new_tokens",
        "kl_risk_threshold",
        "uckv_risk_tolerance",
    ]
    free_by_policy = summarize_free_run(
        free_run,
        [*metadata_cols, "split", "policy", "task_type"],
    )
    free_by_position = summarize_free_run(
        free_run,
        [*metadata_cols, "split", "policy", "answer_position"],
    )
    free_by_context = summarize_free_run(
        free_run,
        [*metadata_cols, "split", "policy", "context_words"],
    )
    tradeoff = make_tradeoff(free_by_policy, replay)
    simulated = simulate_tolerance_sweep(run_dirs, parse_floats(args.tolerances))

    outputs = {
        "free_run_summary.csv": free_by_policy,
        "free_run_by_policy.csv": free_by_policy,
        "free_run_by_position.csv": free_by_position,
        "free_run_by_context.csv": free_by_context,
        "replay_by_policy.csv": replay,
        "retrieval_tradeoff.csv": tradeoff,
        "simulated_uckv_tolerance_sweep.csv": simulated,
    }
    for filename, frame in outputs.items():
        frame.to_csv(outdir / filename, index=False)
        print(f"Wrote {outdir / filename} ({len(frame)} rows)")
    write_markdown(tradeoff, outdir / "retrieval_tradeoff.md")
    print(f"Wrote {outdir / 'retrieval_tradeoff.md'}")
    write_tradeoff_plot(tradeoff, outdir / "retrieval_tradeoff.png")
    if (outdir / "retrieval_tradeoff.png").exists():
        print(f"Wrote {outdir / 'retrieval_tradeoff.png'}")


if __name__ == "__main__":
    main()
