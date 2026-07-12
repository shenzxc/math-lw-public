#!/usr/bin/env python3
"""Aggregate UCKV2 L1 matched-budget experiments.

The script intentionally treats completed run directories as immutable inputs and
writes compact analysis artifacts for paper drafting and follow-up decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


POLICY_ORDER = [
    "full",
    "h2o_hh_96",
    "uckv2_fixed_96",
    "h2o_hh_128",
    "uckv2_fixed_128",
    "h2o_hh_192",
    "uckv2_fixed_192",
    "h2o_hh_256",
    "uckv2_fixed_256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/experiments")
    parser.add_argument("--outdir", default="outputs/analysis_uckv2_l1")
    parser.add_argument("--label-contains", default="b96_256")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_seed(run_label: str) -> str:
    match = re.search(r"seed(\d+)", run_label)
    return match.group(1) if match else "unknown"


def policy_family(policy: str) -> str:
    if policy == "full":
        return "Full KV"
    if policy.startswith("h2o_hh_"):
        return "H2O"
    if policy.startswith("uckv2_fixed_"):
        return "UCKV2"
    return policy


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


def add_metadata(frame: pd.DataFrame, run_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    run_label = str(config.get("run_label", ""))
    out.insert(0, "run_dir", str(run_dir))
    out.insert(1, "run_label", run_label)
    out.insert(2, "seed_id", extract_seed(run_label))
    out.insert(3, "model", config.get("model", ""))
    out.insert(4, "max_prompts", config.get("max_prompts"))
    out.insert(5, "max_new_tokens", config.get("max_new_tokens"))
    out.insert(6, "uckv2_lambda", config.get("uckv2_lambda"))
    out.insert(7, "uckv2_beta", config.get("uckv2_beta"))
    out.insert(8, "uckv2_evict_every", config.get("uckv2_evict_every"))
    return out


def collect_summaries(run_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        config = read_json(run_dir / "config.json")
        frame = pd.read_csv(run_dir / "free_run_summary.csv")
        frame = add_metadata(frame, run_dir, config)
        frame["family"] = frame["policy"].map(policy_family)
        frame["budget"] = frame["policy"].map(policy_budget)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_generations(run_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "free_run_generations.csv"
        if not path.exists():
            continue
        config = read_json(run_dir / "config.json")
        frame = pd.read_csv(path)
        frame = add_metadata(frame, run_dir, config)
        frame["family"] = frame["policy"].map(policy_family)
        frame["budget"] = frame["policy"].map(policy_budget)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def standard_error(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def aggregate_by_policy(summaries: pd.DataFrame) -> pd.DataFrame:
    eval_rows = summaries[summaries["split"].eq("evaluation")].copy()
    if eval_rows.empty:
        return pd.DataFrame()
    grouped = (
        eval_rows.groupby(["policy", "family", "budget"], dropna=False)
        .agg(
            seeds=("seed_id", "nunique"),
            answer_contains=("answer_contains", "mean"),
            answer_contains_se=("answer_contains", standard_error),
            min_answer_contains=("answer_contains", "min"),
            max_answer_contains=("answer_contains", "max"),
            avg_kept_tokens=("avg_kept_tokens", "mean"),
            avg_kept_tokens_se=("avg_kept_tokens", standard_error),
            decode_tokens_per_s=("decode_tokens_per_s", "mean"),
            selector_decode_fraction=("avg_selector_decode_fraction", "mean"),
            avg_eviction_count=("avg_eviction_count", "mean"),
            avg_entropy_gate=("avg_entropy_gate", "mean"),
        )
        .reset_index()
    )
    order = {policy: index for index, policy in enumerate(POLICY_ORDER)}
    grouped["_order"] = grouped["policy"].map(order).fillna(999)
    return grouped.sort_values(["_order", "policy"]).drop(columns="_order")


def budget_deltas(policy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in [96, 128, 192, 256]:
        h2o = policy_summary[policy_summary["policy"].eq(f"h2o_hh_{budget}")]
        uckv2 = policy_summary[policy_summary["policy"].eq(f"uckv2_fixed_{budget}")]
        if h2o.empty or uckv2.empty:
            continue
        h = h2o.iloc[0]
        u = uckv2.iloc[0]
        rows.append(
            {
                "budget": budget,
                "h2o_answer_contains": h["answer_contains"],
                "uckv2_answer_contains": u["answer_contains"],
                "answer_delta": u["answer_contains"] - h["answer_contains"],
                "h2o_avg_kept_tokens": h["avg_kept_tokens"],
                "uckv2_avg_kept_tokens": u["avg_kept_tokens"],
                "kept_delta": u["avg_kept_tokens"] - h["avg_kept_tokens"],
                "h2o_decode_tokens_per_s": h["decode_tokens_per_s"],
                "uckv2_decode_tokens_per_s": u["decode_tokens_per_s"],
                "decode_tokens_per_s_delta": u["decode_tokens_per_s"] - h["decode_tokens_per_s"],
                "uckv2_selector_decode_fraction": u["selector_decode_fraction"],
            }
        )
    return pd.DataFrame(rows)


def exact_mcnemar_p(h2o_only: int, uckv2_only: int) -> float:
    """Two-sided exact McNemar p-value under p=0.5 discordant outcomes."""

    discordant = h2o_only + uckv2_only
    if discordant == 0:
        return 1.0
    observed = min(h2o_only, uckv2_only)
    tail = sum(math.comb(discordant, k) for k in range(observed + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_delta_ci(deltas: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if len(deltas) == 0 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    means = deltas[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_prompt_deltas(
    generations: pd.DataFrame,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    eval_rows = generations[generations["split"].eq("evaluation")].copy()
    rows: list[dict[str, Any]] = []
    for budget in [96, 128, 192, 256]:
        h_policy = f"h2o_hh_{budget}"
        u_policy = f"uckv2_fixed_{budget}"
        subset = eval_rows[eval_rows["policy"].isin([h_policy, u_policy])].copy()
        if subset.empty:
            continue
        pivot = subset.pivot_table(
            index=["seed_id", "prompt_id"],
            columns="policy",
            values="answer_contains",
            aggfunc="first",
        )
        if h_policy not in pivot or u_policy not in pivot:
            continue
        pivot = pivot.dropna(subset=[h_policy, u_policy])
        h = pivot[h_policy].astype(int)
        u = pivot[u_policy].astype(int)
        delta_values = (u - h).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_delta_ci(
            delta_values,
            samples=bootstrap_samples,
            seed=bootstrap_seed + budget,
        )
        h2o_only = int(((h == 1) & (u == 0)).sum())
        uckv2_only = int(((h == 0) & (u == 1)).sum())
        rows.append(
            {
                "budget": budget,
                "paired_prompts": len(pivot),
                "both_correct": int(((h == 1) & (u == 1)).sum()),
                "h2o_only": h2o_only,
                "uckv2_only": uckv2_only,
                "both_wrong": int(((h == 0) & (u == 0)).sum()),
                "paired_answer_delta": float(delta_values.mean()),
                "paired_answer_delta_ci95_low": ci_low,
                "paired_answer_delta_ci95_high": ci_high,
                "mcnemar_exact_p": exact_mcnemar_p(h2o_only, uckv2_only),
            }
        )
    return pd.DataFrame(rows)


def main_takeaway(deltas: pd.DataFrame, paired: pd.DataFrame) -> str:
    if deltas.empty:
        return "No matched-budget H2O versus UCKV2 comparison is available."

    valid = deltas.dropna(subset=["answer_delta"]).copy()
    if valid.empty:
        return "Matched-budget rows were found, but answer deltas are unavailable."

    best = valid.loc[valid["answer_delta"].idxmax()]
    worst = valid.loc[valid["answer_delta"].idxmin()]
    all_negative = bool((valid["answer_delta"] < 0).all())
    all_non_positive = bool((valid["answer_delta"] <= 0).all())
    all_non_negative = bool((valid["answer_delta"] >= 0).all())

    def paired_fragment(budget: int) -> str:
        if paired.empty or "budget" not in paired:
            return ""
        row = paired[paired["budget"].eq(budget)]
        if row.empty:
            return ""
        row = row.iloc[0]
        return (
            f" The paired comparison at budget {budget} has {int(row['paired_prompts'])} "
            f"prompts, H2O-only correct={int(row['h2o_only'])}, "
            f"UCKV2-only correct={int(row['uckv2_only'])}, and McNemar "
            f"p={row['mcnemar_exact_p']:.3g}."
        )

    best_budget = int(best["budget"])
    best_delta = float(best["answer_delta"])
    worst_budget = int(worst["budget"])
    worst_delta = float(worst["answer_delta"])

    if all_negative:
        return (
            "UCKV2 is worse than H2O at every matched budget in this run. "
            f"The least negative point is budget {best_budget} with answer-containment "
            f"delta {best_delta:+.4f}; the worst point is budget {worst_budget} "
            f"with delta {worst_delta:+.4f}. This run should be treated as a failed "
            "confirmation for the current UCKV2 selector, not as evidence of an "
            f"improved quality-memory frontier.{paired_fragment(best_budget)}"
        )
    if all_non_positive:
        return (
            "UCKV2 does not improve over H2O at any matched budget in this run. "
            f"The best point is budget {best_budget} with answer-containment delta "
            f"{best_delta:+.4f}. This is not a positive gate for the current selector."
            f"{paired_fragment(best_budget)}"
        )
    if all_non_negative:
        return (
            f"UCKV2 improves or ties H2O at every matched budget in this run. The "
            f"strongest point is budget {best_budget}, where answer containment changes "
            f"by {best_delta:+.4f} at nearly matched retained-token count."
            f"{paired_fragment(best_budget)}"
        )
    return (
        "UCKV2 has mixed matched-budget behavior in this run. "
        f"The best point is budget {best_budget} with answer-containment delta "
        f"{best_delta:+.4f}; the weakest point is budget {worst_budget} with delta "
        f"{worst_delta:+.4f}. Positive local gates should therefore be confirmed on "
        f"larger models before being used as a paper claim.{paired_fragment(best_budget)}"
    )


def write_markdown(
    policy_summary: pd.DataFrame,
    deltas: pd.DataFrame,
    paired: pd.DataFrame,
    outpath: Path,
) -> None:
    if policy_summary.empty:
        outpath.write_text("No completed UCKV2 L1 runs found.\n", encoding="utf-8")
        return

    display = policy_summary.copy()
    for col in [
        "answer_contains",
        "answer_contains_se",
        "avg_kept_tokens",
        "decode_tokens_per_s",
        "selector_decode_fraction",
    ]:
        if col in display:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    delta_display = deltas.copy()
    for col in [
        "h2o_answer_contains",
        "uckv2_answer_contains",
        "answer_delta",
        "h2o_avg_kept_tokens",
        "uckv2_avg_kept_tokens",
        "kept_delta",
        "decode_tokens_per_s_delta",
        "uckv2_selector_decode_fraction",
    ]:
        if col in delta_display:
            delta_display[col] = delta_display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    text = [
        "# UCKV2 L1 Matched-Budget Results",
        "",
        "Evaluation split only. Scores are averaged over completed seeds.",
        "",
        "## Policy Summary",
        "",
        display[
            [
                "policy",
                "seeds",
                "answer_contains",
                "answer_contains_se",
                "min_answer_contains",
                "max_answer_contains",
                "avg_kept_tokens",
                "decode_tokens_per_s",
                "selector_decode_fraction",
            ]
        ].to_markdown(index=False),
        "",
        "## Matched-Budget Deltas",
        "",
        delta_display.to_markdown(index=False),
        "",
        "## Paired Prompt Counts",
        "",
        paired.to_markdown(index=False, floatfmt=".6g") if not paired.empty else "No paired prompt rows found.",
        "",
        "## Main Takeaway",
        "",
        main_takeaway(deltas, paired),
        "",
    ]
    outpath.write_text("\n".join(text), encoding="utf-8")


def write_plot(policy_summary: pd.DataFrame, outpath: Path) -> None:
    if policy_summary.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = policy_summary[policy_summary["policy"].ne("full")].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    markers = {"H2O": "o", "UCKV2": "s"}
    colors = {"H2O": "#4C78A8", "UCKV2": "#F58518"}
    for family, family_rows in rows.groupby("family"):
        family_rows = family_rows.sort_values("avg_kept_tokens")
        ax.errorbar(
            family_rows["avg_kept_tokens"],
            family_rows["answer_contains"],
            yerr=family_rows["answer_contains_se"],
            marker=markers.get(family, "o"),
            color=colors.get(family, None),
            linewidth=1.8,
            capsize=3,
            label=family,
        )
        for _, row in family_rows.iterrows():
            ax.annotate(
                str(int(row["budget"])),
                (row["avg_kept_tokens"], row["answer_contains"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
    full = policy_summary[policy_summary["policy"].eq("full")]
    if not full.empty:
        ax.axhline(
            full["answer_contains"].iloc[0],
            color="#54A24B",
            linestyle="--",
            linewidth=1.2,
            label="Full KV",
        )
    ax.set_xlabel("Average retained KV tokens")
    ax.set_ylabel("Answer containment")
    ax.set_title("UCKV2 L1 matched-budget tradeoff")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_run_dirs(root, args.label_contains)
    summaries = collect_summaries(run_dirs)
    generations = collect_generations(run_dirs)
    policy_summary = aggregate_by_policy(summaries)
    deltas = budget_deltas(policy_summary)
    paired = paired_prompt_deltas(
        generations,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    summaries.to_csv(outdir / "seed_policy_summary.csv", index=False)
    generations.to_csv(outdir / "free_run_generations.csv", index=False)
    policy_summary.to_csv(outdir / "policy_summary.csv", index=False)
    deltas.to_csv(outdir / "matched_budget_deltas.csv", index=False)
    paired.to_csv(outdir / "paired_prompt_counts.csv", index=False)
    write_markdown(policy_summary, deltas, paired, outdir / "uckv2_l1_results.md")
    write_plot(policy_summary, outdir / "uckv2_l1_tradeoff.png")

    print(f"Run dirs: {len(run_dirs)}")
    print(f"Wrote {outdir}")
    if not deltas.empty:
        print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
