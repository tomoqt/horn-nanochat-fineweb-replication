#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCALING_RUNS = ROOT / "scaling_runs"
OUTDIR = SCALING_RUNS / "plots"

MODEL_BRANCH_SUMMARY = SCALING_RUNS / "model_branch" / "model_branch_summary.json"
HORIZON_BRANCH_SUMMARY = SCALING_RUNS / "horizon_branch" / "horizon_branch_summary.json"
JOINT_BRANCH_SUMMARY = SCALING_RUNS / "joint_branch" / "joint_branch_summary.json"
JOINT_BENCHMARK_SUMMARY = (
    SCALING_RUNS
    / "joint_branch"
    / "joint_scaled_from_branch_winners"
    / "benchmark_summary.json"
)

COLORS = {"baseline": "#1f77b4", "horn": "#d62728"}
VARIANTS = ("baseline", "horn")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def curve_stats(runs: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_step: dict[int, list[float]] = {}
    for run in runs:
        for step, loss in zip(run["val_steps"], run["val_losses"]):
            by_step.setdefault(int(step), []).append(float(loss))
    steps = np.array(sorted(by_step), dtype=float)
    means = np.array([np.mean(by_step[int(s)]) for s in steps], dtype=float)
    stds = np.array([np.std(by_step[int(s)]) for s in steps], dtype=float)
    return steps, means, stds


def plot_model_size_curves() -> Path:
    data = load_json(MODEL_BRANCH_SUMMARY)
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in data["per_run_results"]:
        exp = row["experiment"]
        variant = row["variant"]
        run_data = load_json(Path(row["path"]))
        run = run_data["runs"][0]
        grouped.setdefault(exp, {}).setdefault(variant, []).append(run)

    order = ["small_fixed_horizon", "medium_fixed_horizon", "large_fixed_horizon"]
    titles = {
        "small_fixed_horizon": "Small",
        "medium_fixed_horizon": "Medium",
        "large_fixed_horizon": "Large",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, exp in zip(axes, order):
        exp_data = grouped[exp]
        for variant in VARIANTS:
            runs = exp_data[variant]
            for run in runs:
                ax.plot(
                    run["val_steps"],
                    run["val_losses"],
                    color=COLORS[variant],
                    alpha=0.22,
                    linewidth=1.2,
                )
            steps, means, stds = curve_stats(runs)
            ax.plot(
                steps,
                means,
                color=COLORS[variant],
                linewidth=2.4,
                label=f"{variant} (final {means[-1]:.4f})",
            )
            ax.fill_between(
                steps,
                means - stds,
                means + stds,
                color=COLORS[variant],
                alpha=0.15,
                linewidth=0,
            )
        ax.set_title(f"Model Size: {titles[exp]}")
        ax.set_xlabel("Eval Step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Validation Loss")
    axes[-1].legend(loc="upper right", frameon=True)
    fig.suptitle("FineWeb Model-Size Branch: Validation-Loss Curves", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = OUTDIR / "model_size_val_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_horizon_final_bars() -> Path:
    data = load_json(HORIZON_BRANCH_SUMMARY)
    horizons = data["horizons"]

    labels = [f"{h['block_size']}" for h in horizons]
    baseline = np.array([h["baseline_final_val_loss_mean"] for h in horizons], dtype=float)
    horn = np.array([h["horn_final_val_loss_mean"] for h in horizons], dtype=float)
    delta = horn - baseline

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - width / 2, baseline, width, label="baseline", color=COLORS["baseline"])
    ax.bar(x + width / 2, horn, width, label="horn", color=COLORS["horn"])
    for i, d in enumerate(delta):
        ax.text(
            i,
            max(baseline[i], horn[i]) + 0.004,
            f"Δ={d:+.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x, labels)
    ax.set_xlabel("Token Horizon (block_size)")
    ax.set_ylabel("Final Validation Loss (mean over seeds)")
    ax.set_title("FineWeb Token-Horizon Branch: Baseline vs HORN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = OUTDIR / "horizon_final_loss_bars.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_joint_curves() -> Path:
    bench = load_json(JOINT_BENCHMARK_SUMMARY)
    joint = load_json(JOINT_BRANCH_SUMMARY)
    by_variant: dict[str, list[dict[str, Any]]] = {k: [] for k in VARIANTS}
    for run in bench["runs"]:
        by_variant[run["variant"]].append(run)

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    for variant in VARIANTS:
        runs = by_variant[variant]
        for run in runs:
            ax.plot(
                run["val_steps"],
                run["val_losses"],
                color=COLORS[variant],
                alpha=0.22,
                linewidth=1.1,
            )
        steps, means, stds = curve_stats(runs)
        ax.plot(
            steps,
            means,
            color=COLORS[variant],
            linewidth=2.5,
            label=f"{variant} (final {means[-1]:.4f})",
        )
        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            color=COLORS[variant],
            alpha=0.16,
            linewidth=0,
        )

    delta = float(joint["summary"]["delta_final_vs_baseline"])
    rel = float(joint["summary"]["relative_improvement_final_pct"])
    ax.set_title("FineWeb Joint Branch: Validation-Loss Curves")
    ax.set_xlabel("Eval Step")
    ax.set_ylabel("Validation Loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    ax.text(
        0.02,
        0.02,
        f"Final delta (horn-baseline): {delta:+.4f} ({rel:+.3f}%)",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )
    fig.tight_layout()

    out = OUTDIR / "joint_val_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_compact_metrics() -> Path:
    model = load_json(MODEL_BRANCH_SUMMARY)
    horizon = load_json(HORIZON_BRANCH_SUMMARY)
    joint = load_json(JOINT_BRANCH_SUMMARY)

    rows: list[dict[str, Any]] = []

    model_group: dict[str, dict[str, float]] = {}
    for r in model["aggregate_results"]:
        model_group.setdefault(r["experiment"], {})[r["variant"]] = float(
            r["final_val_loss_mean"]
        )
    for exp, vals in sorted(model_group.items()):
        b = vals["baseline"]
        h = vals["horn"]
        rows.append(
            {
                "group": "model_size",
                "name": exp,
                "baseline_final_val_loss_mean": b,
                "horn_final_val_loss_mean": h,
                "delta_horn_minus_baseline": h - b,
            }
        )

    for h in horizon["horizons"]:
        rows.append(
            {
                "group": "token_horizon",
                "name": h["name"],
                "baseline_final_val_loss_mean": float(h["baseline_final_val_loss_mean"]),
                "horn_final_val_loss_mean": float(h["horn_final_val_loss_mean"]),
                "delta_horn_minus_baseline": float(h["horn_delta_final_vs_baseline"]),
            }
        )

    rows.append(
        {
            "group": "joint",
            "name": "joint_scaled_from_branch_winners",
            "baseline_final_val_loss_mean": float(
                joint["summary"]["baseline"]["final_val_loss_mean"]
            ),
            "horn_final_val_loss_mean": float(joint["summary"]["horn"]["final_val_loss_mean"]),
            "delta_horn_minus_baseline": float(joint["summary"]["delta_final_vs_baseline"]),
        }
    )

    out = OUTDIR / "loss_plot_metrics.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    outputs = [
        plot_model_size_curves(),
        plot_horizon_final_bars(),
        plot_joint_curves(),
        write_compact_metrics(),
    ]
    print("\n".join(str(p) for p in outputs))


if __name__ == "__main__":
    main()
