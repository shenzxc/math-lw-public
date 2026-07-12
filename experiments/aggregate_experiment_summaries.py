#!/usr/bin/env python3
"""Aggregate UCKV pilot run outputs into compact CSV/Markdown summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--outdir", default="outputs/analysis")
    parser.add_argument("--label-contains", default="")
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(root: Path, label_contains: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for config_path in sorted(root.glob("*/config.json")):
        run_dir = config_path.parent
        summary_path = run_dir / "summary.csv"
        risk_path = run_dir / "risk_summary.csv"
        selection_path = run_dir / "uckv_selection_summary.csv"
        if not summary_path.exists():
            continue

        config = read_json(config_path)
        run_label = str(config.get("run_label", ""))
        if label_contains and label_contains not in run_label:
            continue

        summary = pd.read_csv(summary_path)
        risk_summary = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()
        selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()

        for _, row in summary.iterrows():
            out = {
                "run_dir": str(run_dir),
                "run_label": run_label,
                "model": config.get("model"),
                "split_mode": config.get("split_mode"),
                "max_prompts": config.get("max_prompts"),
                "max_new_tokens": config.get("max_new_tokens"),
                "kl_risk_threshold": config.get("kl_risk_threshold"),
                "uckv_risk_tolerance": config.get("uckv_risk_tolerance"),
                "policy": row["policy"],
                "split": row["split"],
                "prompts": row["prompts"],
                "steps": row["steps"],
                "avg_kl": row["avg_kl"],
                "p95_kl": row["p95_kl"],
                "max_kl": row["max_kl"],
                "top1_mismatch_rate": row["top1_mismatch_rate"],
                "avg_retained_ratio": row["avg_retained_ratio"],
                "avg_kept_tokens": row["avg_kept_tokens"],
                "avg_full_tokens": row["avg_full_tokens"],
                "avg_replay_s_per_step": row["avg_replay_s_per_step"],
            }
            risk_eval = risk_summary[risk_summary.get("split", pd.Series(dtype=str)).eq(row["split"])]
            if len(risk_eval):
                risk_row = risk_eval.iloc[0]
                out.update(
                    {
                        "risk_positive_rate": risk_row.get("positive_rate"),
                        "risk_brier": risk_row.get("brier"),
                        "risk_ece_10": risk_row.get("ece_10"),
                        "risk_roc_auc": risk_row.get("roc_auc"),
                    }
                )
            if row["policy"] == "uckv_budget" and len(selection):
                sel_split = selection[selection.get("split", pd.Series(dtype=str)).eq(row["split"])]
                out["uckv_fallback_steps"] = int(
                    sel_split[sel_split.get("uckv_selection", pd.Series(dtype=str)).eq("fallback_max_budget")]
                    .get("steps", pd.Series(dtype=int))
                    .sum()
                )
                out["uckv_min_safe_steps"] = int(
                    sel_split[sel_split.get("uckv_selection", pd.Series(dtype=str)).eq("min_safe_budget")]
                    .get("steps", pd.Series(dtype=int))
                    .sum()
                )
            rows.append(out)
    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, outpath: Path) -> None:
    if df.empty:
        outpath.write_text("No completed runs found.\n", encoding="utf-8")
        return

    eval_rows = df[df["split"].eq("evaluation")].copy()
    key_mask = (
        eval_rows["policy"].eq("uckv_budget")
        | eval_rows["policy"].eq("uckv_entropy_adaptive")
        | eval_rows["policy"].str.startswith("sink4_window_")
    )
    eval_rows = eval_rows[key_mask]
    cols = [
        "run_label",
        "kl_risk_threshold",
        "uckv_risk_tolerance",
        "policy",
        "avg_retained_ratio",
        "avg_kl",
        "top1_mismatch_rate",
        "risk_roc_auc",
        "uckv_fallback_steps",
    ]
    existing = [col for col in cols if col in eval_rows.columns]
    text = [
        "# Sweep Summary",
        "",
        "Evaluation split, selected policies only.",
        "",
        eval_rows[existing].sort_values(["run_label", "policy"]).to_markdown(index=False),
        "",
    ]
    outpath.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = collect_runs(root, args.label_contains)
    csv_path = outdir / "experiment_summary.csv"
    md_path = outdir / "experiment_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
